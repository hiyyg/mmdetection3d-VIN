import argparse
import json
import pickle
from bisect import bisect
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pcl
from addict import Dict as edict
from d3d.dataset.nuscenes import NuscenesSegmentationClass
from d3d.vis.pcl import visualize_detections
from pcl.visualization import RenderingProperties
from PIL import Image
from tqdm import tqdm, trange
from subprocess import check_call, DEVNULL

parser = argparse.ArgumentParser()
parser.add_argument("dump_folder", help="Path to the archive for visualization", type=str)
parser.add_argument("initial_idx", help="Specify the initial frame index", type=int)
parser.add_argument("-b", "--show-box", help="Show box at beginning", action="store_true")
parser.add_argument("-c", "--show-box-compare", help="Show GT box along with predictions", action="store_true")
parser.add_argument("-t", "--show-text", help="Show text labels", action="store_true")
parser.add_argument("-p", "--point-size", help="Specify default point size", type=int, default=3)
parser.add_argument("-r", "--record-path", help="Path to record output path", type=str, default="record")
parser.add_argument("-i", "--record-interval", help="Interval (in seconds) between record frames", type=float, default=0.5)
parser.add_argument("-w", "--save-webp", help="Save animate file as webp files", action="store_true")
parser.add_argument("-l", "--highlight", help="Highlight objects with spcified index", type=int, nargs='+')
parser.add_argument("-s", "--score-threshold", help="Only show objects with score higher than this number", type=float, default=0)
args = parser.parse_args()

visualizer = None
def key_callback(event):
    global visualizer
    visualizer.key_handler(event)

class ResultVisualizer():
    def __init__(self, dump_folder: str, initial_idx=0, initial_configs:dict = None) -> None:
        if dump_folder.endswith(".zip") and Path(dump_folder).exists():
            self.dump_folder = ZipFile(dump_folder, "r")
            self.inzip = True
        else:
            self.dump_folder = Path(dump_folder)
            self.inzip = False

        self.configs = edict(
            point_size=3,
            show_box=False,
            show_box_compare=False,
            show_text=False,
            record_path="record",
            record_interval=0.5,
            score_threshold=0,
            highlight=set(),
            save_webp=False)
        self.configs.update(initial_configs)
        self.configs.highlight = self.configs.highlight or set()
        self.vis = pcl.Visualizer()

        self.legend_ratio = 0.1
        self.has_gt = None
        self.scan_folder(initial_idx)
        self.render_frame(self.current_seq, self.current_fidx)

    def scan_folder(self, initial_idx):
        if self.inzip:
            idlist = json.loads(self.dump_folder.read("idlist.json"))
        else:
            idlist = json.loads(Path(self.dump_folder, "idlist.json").read_bytes())

        mapping = defaultdict(dict)
        for idx, (scene, fid) in enumerate(idlist):
            mapping[scene][fid] = idx
        self.idx_mapping = {k: [v[f] for f in range(max(v)+1)] for k, v in mapping.items()}

        self.current_seq, self.current_fidx = idlist[initial_idx]

    def init_renderer(self):
        if self.has_gt is not None:
            return
        self.has_gt = "anno_gt" in self.current_data

        # register callback
        self.vis.setCameraPosition([0,0,50], [0,1,1])
        self.vis.registerKeyboardCallback(key_callback)

        if self.has_gt:
            # create viewports
            self.vp_gt = self.vis.createViewPort(0, 0, 0.5, 1 - self.legend_ratio)
            self.vp_pred = self.vis.createViewPort(0.5, 0, 1, 1 - self.legend_ratio)
            self.vp_legend = self.vis.createViewPort(0, 1 - self.legend_ratio, 1, 1)

            # draw legend
            self.current_data.semantic_colormap[0] = (255, 255, 255) # assign white to ignore
            cmap = np.array(self.current_data.semantic_colormap) / 255.
            fontsize = 28
            lx, ly = 0, 0
            for i, c in enumerate(cmap):
                text = NuscenesSegmentationClass(i).name
                self.vis.addText(text, lx, ly, color=c, fontsize=fontsize, id="legend%d" % i, viewport=self.vp_legend)
                lx += len(text) * fontsize * 0.7
                if lx > 1000:
                    ly += fontsize
                    lx = 0
        else:
            # create viewports
            self.vp_pred = self.vis.createViewPort(0, 0, 1, 1 - self.legend_ratio)
            self.vp_legend = self.vis.createViewPort(0, 1 - self.legend_ratio, 1, 1)

    def draw_clouds(self):
        # draw point cloud
        if self.has_gt:
            if 'semantic_dt' in self.current_data:
                cloud_gt = np.empty(len(self.current_data.cloud), dtype=[
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4'), 
                    ('semantic', 'u1')
                ])
                cloud_gt['x'] = self.current_data.cloud[:, 0]
                cloud_gt['y'] = self.current_data.cloud[:, 1]
                cloud_gt['z'] = self.current_data.cloud[:, 2]
                cloud_gt['intensity'] = self.current_data.cloud[:, 3]
                cloud_gt['semantic'] = self.current_data.semantic_gt
                cloud_gt = pcl.PointCloud(cloud_gt)

                cloud_pred = np.empty(len(self.current_data.cloud), dtype=[
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4'), 
                    ('semantic', 'u1'), ('correct', 'u1'), ('score', 'f4')
                ])
                cloud_pred['x'] = self.current_data.cloud[:, 0]
                cloud_pred['y'] = self.current_data.cloud[:, 1]
                cloud_pred['z'] = self.current_data.cloud[:, 2]
                cloud_pred['intensity'] = self.current_data.cloud[:, 3]
                cloud_pred['semantic'] = self.current_data.semantic_dt
                cloud_pred['correct'] = (self.current_data.semantic_gt == 0) | (self.current_data.semantic_gt == self.current_data.semantic_dt)
                cloud_pred['score'] = self.current_data.semantic_scores
                cloud_pred = pcl.PointCloud(cloud_pred)

                cmap = np.array(self.current_data.semantic_colormap)
                cmap_handler = lambda cloud: cmap[cloud['semantic']]

                self.vis.addPointCloud(cloud_gt, color_handler=cmap_handler, viewport=self.vp_gt, id="cloud_gt")
                self.vis.addPointCloud(cloud_gt, field='intensity', viewport=self.vp_gt, id="cloud_gt")
                self.vis.addPointCloud(cloud_pred, color_handler=cmap_handler, viewport=self.vp_pred, id="cloud_pred")
                self.vis.addPointCloud(cloud_pred, field='intensity', viewport=self.vp_pred, id="cloud_pred")
                self.vis.addPointCloud(cloud_pred, field='correct', viewport=self.vp_pred, id="cloud_pred")
                self.vis.addPointCloud(cloud_pred, field='score', viewport=self.vp_pred, id="cloud_pred")

                self.vis.setPointCloudRenderingProperties(RenderingProperties.PointSize, self.configs.point_size, "cloud_gt")
                self.vis.setPointCloudRenderingProperties(RenderingProperties.PointSize, self.configs.point_size, "cloud_pred")
            else:
                self.vis.addPointCloud(pcl.create_xyzi(self.current_data.cloud[:,:4]), id="cloud")
        else:
            if 'semantic_dt' in self.current_data:
                cmap = np.array(self.current_data.semantic_colormap)
                cmap_handler = lambda cloud: cmap[cloud['label']]

                self.vis.addPointCloud(pcl.create_xyzl(np.hstack([self.current_data.cloud[:,:3], self.current_data.semantic_dt.reshape(-1,1)])), color_handler=cmap_handler, id="cloud")
            else:
                self.vis.addPointCloud(pcl.create_xyzi(self.current_data.cloud[:,:4]), id="cloud")

    def draw_boxes(self):
        anno_dt = self.current_data.anno_dt.filter_score(self.configs.score_threshold)
        if self.configs.show_box:
            text_scale = 0.8 if self.configs.show_text else 0
            if self.has_gt:
                visualize_detections(self.vis, self.current_data.anno_gt.frame, self.current_data.anno_gt, None, text_scale=text_scale, id_prefix="gt_left", viewport=self.vp_gt)
                if self.configs.show_box_compare:
                    visualize_detections(self.vis, self.current_data.anno_gt.frame, self.current_data.anno_gt, None, text_scale=0, id_prefix="gt_right", viewport=self.vp_pred)
                visualize_detections(self.vis, anno_dt.frame, anno_dt, None, text_scale=text_scale, box_color=(1,1,0), text_color=(1,0.8,0), id_prefix="pred", viewport=self.vp_pred)
                self.vis.setRepresentationToWireframeForAllActors()
            else:
                visualize_detections(self.vis, anno_dt.frame, anno_dt, None, text_scale=text_scale, box_color=(1,1,0), text_color=(1,0.8,0), id_prefix="pred", viewport=self.vp_pred)

            # highlight box
            for i in self.configs.highlight:
                if i < len(anno_dt):
                    self.vis.setShapeRenderingProperties(RenderingProperties.Color, (1.,0.,0.), "pred/target%d" % i)
        else:
            self.vis.removeAllShapes(self.vp_gt)
            self.vis.removeAllShapes(self.vp_pred)

    def render_frame(self, scene, fidx, verbose=True):
        idx = self.idx_mapping[scene][fidx]
        if self.inzip:
            with self.dump_folder.open("%06d.pkl" % idx) as fin:
                self.current_data = edict(pickle.load(fin))
        else:
            with open(self.dump_folder / ("%06d.pkl" % idx), "rb") as fin:
                self.current_data = edict(pickle.load(fin))
        self.current_data.semantic_colormap[0] = (255, 255, 255) # assign white to ignore

        self.init_renderer()

        self.vis.removeAllShapes(viewport=self.vp_gt)
        self.vis.removeAllPointClouds(viewport=self.vp_gt)
        self.vis.removeAllShapes(viewport=self.vp_pred)
        self.vis.removeAllPointClouds(viewport=self.vp_pred)

        self.draw_clouds()
        self.draw_boxes()
        if verbose:
            print("Loaded %s @ %d" % (self.current_seq, self.current_fidx))

    def _save_color_handler(self):
        self._hcolor_gt = self.vis.getColorHandlerIndex("cloud_gt")
        self._hcolor_pred = self.vis.getColorHandlerIndex("cloud_pred")

    def _restore_color_handler(self):
        self.vis.updateColorHandlerIndex("cloud_gt", self._hcolor_gt)
        self.vis.updateColorHandlerIndex("cloud_pred", self._hcolor_pred)
        self.vis.render()

    def key_handler(self, event):
        if not event.keyUp():
            return
        if event.KeyCode == 'b':
            self.configs.show_box = not self.configs.show_box
            self.draw_boxes()
            self.vis.render()
            return
        elif event.KeyCode == 'z':
            print("Start recording sequence")
            self.record_sequence(self.current_seq)
            return
        elif event.KeyCode == 'x':
            print("Recording all sequences..")
            self.record_all()
            return
        elif event.KeyCode == '+':
            self.configs.point_size += 1
        elif event.KeyCode == '-':
            self.configs.point_size = max(1, self.configs.point_size-1)
            
        # following operations requires rendering
        elif event.KeySym in ["Up", "Down", "Left", "Right"]:
            if event.KeySym == "Down":
                if self.current_fidx == 0:
                    return
                self.current_fidx -= 1
            elif event.KeySym == "Up":
                if self.current_fidx >= len(self.idx_mapping[self.current_seq]) - 1:
                    return
                self.current_fidx += 1
            elif event.KeySym == "Left":
                scenes = sorted(self.idx_mapping)
                new_sidx = bisect(scenes, self.current_seq) - 2
                self.current_seq = scenes[max(new_sidx, 0)]
                self.current_fidx = min(self.current_fidx, len(self.idx_mapping[self.current_seq])-1)
            elif event.KeySym == "Right":
                scenes = sorted(self.idx_mapping)
                new_sidx = bisect(scenes, self.current_seq)
                self.current_seq = scenes[min(new_sidx, len(scenes)-1)]
                self.current_fidx = min(self.current_fidx, len(self.idx_mapping[self.current_seq])-1)
            self._save_color_handler()
            self.render_frame(self.current_seq, self.current_fidx)
            self._restore_color_handler()

    def record_sequence(self, seq):
        record_path = Path(self.configs.record_path)
        record_path.mkdir(exist_ok=True, parents=True)
        tmp_img = Path(record_path, "tmp.png").resolve()

        try:
            check_call("gifsicle --version", shell=True, stdout=DEVNULL)
            has_gifsicle = True
        except:
            has_gifsicle = False

        images = []
        self._save_color_handler()
        for fidx in trange(len(self.idx_mapping[seq]), leave=False):
            self.render_frame(seq, fidx, verbose=False)
            self._restore_color_handler()
            self.vis.saveScreenshot(str(tmp_img))
            capture = Image.open(tmp_img)
            if has_gifsicle:
                capture.save(Path(record_path, "tmp%02d.gif" % fidx))
                images.append("tmp%02d.gif" % fidx)
            else:
                capture.load()
                images.append(capture)

        if self.configs.save_webp:
            out_file = Path(record_path, 'record-%s.webp' % seq)
            images[0].save(
                out_file,
                save_all=True,
                append_images=images[1:],
                quality=80,
                duration=int(self.configs.record_interval * 1e3),
                loop=0
            )
        elif has_gifsicle:
            out_file = Path(record_path, 'record-%s.gif' % seq)
            check_call("gifsicle -d%d -l -k%d -O tmp*.gif -o record-%s.gif" % (
                int(self.configs.record_interval * 100), 32, seq
            ), shell=True, cwd=record_path)
            for img in record_path.glob("tmp*.gif"):
                img.unlink()
        else:
            out_file = Path(record_path, 'record-%s.gif' % seq)
            images[0].save(
                out_file,
                save_all=True,
                append_images=images[1:],
                optimize=True,
                duration=int(self.configs.record_interval * 1e3),
                loop=0
            )

        tmp_img.unlink()
        print("Recorded image created at %s" % str(out_file))

    def record_all(self):
        for seq in tqdm(list(self.idx_mapping)):
            self.record_sequence(seq)

    def show(self):
        self.vis.spin()

visualizer = ResultVisualizer(args.dump_folder, args.initial_idx, dict(
    point_size=args.point_size,
    show_box=args.show_box,
    show_box_compare=args.show_box_compare,
    show_text=args.show_text,
    record_path=args.record_path,
    record_interval=args.record_interval,
    score_threshold=args.score_threshold,
    highlight=args.highlight,
    save_webp=args.save_webp
))
visualizer.show()
