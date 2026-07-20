import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from models.lspm import LSPM
from models.common import (
    GhostConv3D,
    GlobalAnchorDetector,
    Encoder,
    MultiScaleSkipFusion,
    Decoder,
    BoundaryRefinement,
)


# LightMedSeg with additional boundary refinement module for greater precision
class LMSBR(Module):
    def __init__(
        self,
        n_classes=2,
        num_anchors=8,
        stage_channels=(8, 16, 32, 64),
        br_hidden_channels=16,
        metadata_film=True,
    ):
        super().__init__()

        self._hyperparams = {
            "n_classes": n_classes,
            "num_anchors": num_anchors,
            "stage_channels": stage_channels,
            "br_hidden_channels": br_hidden_channels,
            "metadata_film": metadata_film
        }

        self.base = LightMedSeg(
            n_classes=n_classes,
            in_channels=1,
            num_anchors=num_anchors,
            stage_channels=stage_channels,
            metadata_film=metadata_film,
            downsample=True,
        )

        self.br = BoundaryRefinement(hidden_channels=br_hidden_channels)
    
    @staticmethod
    def load(path, device=None):
        checkpoint = torch.load(path, map_location=device)
        hparams = checkpoint["hyperparams"]
        model = LMSBR(
            n_classes=hparams["n_classes"],
            num_anchors=hparams["num_anchors"],
            stage_channels=hparams["stage_channels"],
            br_hidden_channels=hparams["br_hidden_channels"],
            metadata_film=hparams["metadata_film"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save(self, path, metadata=None):
        save_dict = {
            "model": "refined",
            "metadata": metadata,
            "model_state_dict": self.state_dict(),
            "hyperparams": self.hyperparams(),
        }
        torch.save(save_dict, path)
    
    @staticmethod
    def small(n_classes=2, metadata_film=True):
        return LMSBR(
            n_classes=n_classes,
            num_anchors=8,
            stage_channels=(8, 16, 32, 64),
            br_hidden_channels=4,
            metadata_film=metadata_film,
        )
        
    @staticmethod
    def medium(n_classes=2, metadata_film=True):
        return LMSBR(
            n_classes=n_classes,
            num_anchors=16,
            stage_channels=(8, 16, 64, 128),
            br_hidden_channels=8,
            metadata_film=metadata_film,
        )
        
    @staticmethod
    def large(n_classes=2, metadata_film=True):
        return LMSBR(
            n_classes=n_classes,
            num_anchors=32,
            stage_channels=(16, 32, 64, 256),
            br_hidden_channels=16,
            metadata_film=metadata_film,
        )
    

    def hyperparams(self):
        return self._hyperparams

    def forward_train(self, X, metadata):
        # original = X[:, 0:1, :, :, :]
        # edges = X[:, 1:4, :, :, :]
        coarse = self.base(X, metadata)
        refined = self.br(coarse, X)
        return refined, coarse

    def forward(self, X, metadata):
        return self.forward_train(X, metadata)[0]


class LightMedSeg(Module):
    def __init__(
        self,
        n_classes=2,
        in_channels=1,
        num_anchors=8,
        stage_channels=(8, 16, 32, 64),
        metadata_film=True,
        boundary_refine=False,
        downsample=True,
    ):
        super().__init__()

        if boundary_refine and not downsample:
            raise ValueError("Boundary refinement requires downsampling.")

        self._hyperparams = {
            "n_classes": n_classes,
            "in_channels": in_channels,
            "num_anchors": num_anchors,
            "metadata_film": metadata_film,
            "downsample": downsample,
            "boundary_refine": boundary_refine
        }

        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.metadata_film = metadata_film
        
        
        # if boundary_refine:
        #     self.boundary_refinement = BoundaryRefinement(in_channels=)

        self.embedding_stem = GhostConv3D(in_channels, 8, downscale=downsample)

        self.anchor_detector = GlobalAnchorDetector(8, 8, num_anchors=num_anchors)
        self.lspm = LSPM()
        self.E1 = Encoder(
            8,
            stage_channels[0],
            num_anchors=num_anchors,
            metadata_film=metadata_film,
            downsample=True,
        )
        self.E2 = Encoder(
            stage_channels[0],
            stage_channels[1],
            num_anchors=num_anchors,
            metadata_film=metadata_film,
            downsample=True,
        )
        self.E3 = Encoder(
            stage_channels[1],
            stage_channels[2],
            num_anchors=num_anchors,
            metadata_film=metadata_film,
            downsample=True,
        )
        self.E4 = Encoder(
            stage_channels[2],
            stage_channels[3],
            num_anchors=num_anchors,
            metadata_film=metadata_film,
            downsample=False,
        )
        self.skip_fusion = MultiScaleSkipFusion(stage_channels=stage_channels)
        self.D1 = Decoder(
            stage_channels[3],
            stage_channels[2],
            num_anchors=num_anchors,
            upsample=False,
        )
        self.D2 = Decoder(
            stage_channels[2], stage_channels[1], num_anchors=num_anchors, upsample=True
        )
        self.D3 = Decoder(
            stage_channels[1], stage_channels[0], num_anchors=num_anchors, upsample=True
        )
        self.D4 = Decoder(stage_channels[0], 8, num_anchors=num_anchors, upsample=True)
        self.segmentation_head = nn.Conv3d(
            8, n_classes, kernel_size=1, stride=1, padding=0
        )
        # self.final_upsample = nn.ConvTranspose3d(
        #     n_classes, n_classes, kernel_size=2, stride=2
        # )
        
    @staticmethod
    def load(path, device=None):
        checkpoint = torch.load(path, map_location=device)
        hparams = checkpoint["hyperparams"]
        model = LightMedSeg(
            n_classes=hparams["n_classes"],
            in_channels=hparams["in_channels"],
            num_anchors=hparams["num_anchors"],
            stage_channels=hparams["stage_channels"],
            metadata_film=hparams["metadata_film"],
            downsample=hparams['downsample']
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    
    def save(self, path, metadata=None):
        save_dict = {
            "model": "base",
            "metadata": metadata,
            "model_state_dict": self.state_dict(),
            "hyperparams": self.hyperparams(),
        }
        torch.save(save_dict, path)

    @staticmethod
    def small(n_classes=2, in_channels=1, metadata_film=True, downsample=True):
        return LightMedSeg(
            n_classes=n_classes,
            in_channels=in_channels,
            num_anchors=8,
            stage_channels=(8, 16, 32, 64),
            metadata_film=metadata_film,
            downsample=downsample
        )
        
    @staticmethod
    def medium(n_classes=2, in_channels=1, metadata_film=True, downsample=True):
        return LightMedSeg(
            n_classes=n_classes,
            in_channels=in_channels,
            num_anchors=16,
            stage_channels=(8, 16, 64, 128),
            metadata_film=metadata_film,
            downsample=downsample
        )
        
    @staticmethod
    def large(n_classes=2, in_channels=1, metadata_film=True, downsample=True):
        return LightMedSeg(
            n_classes=n_classes,
            in_channels=in_channels,
            num_anchors=32,
            stage_channels=(16, 32, 64, 256),
            metadata_film=metadata_film,
            downsample=downsample
        )

    def hyperparams(self):
        return self._hyperparams

    def forward(self, X, metadata):
        _, _, D, H, W = X.shape
        embedding = self.embedding_stem(X)
        anchors = self.anchor_detector(embedding)
        T, f0 = self.lspm(embedding)
        # print(T, f0, anchors)
        e1_out, E1 = self.E1(f0, anchors, T, metadata)
        # print(e1_out.shape)
        e2_out, E2 = self.E2(e1_out, anchors, T, metadata)
        # del e1_out
        e3_out, E3 = self.E3(e2_out, anchors, T, metadata)
        # del e2_out
        e4_out, E4 = self.E4(e3_out, anchors, T, metadata)
        # del e3_out
        skip_1, skip_2, skip_3, skip_4 = self.skip_fusion(E1, E2, E3, E4)
        # del E1, E2, E3, E4
        # print(E1.shape, E2.shape, E3.shape, E4.shape)
        # print(skip_1.shape, skip_2.shape, skip_3.shape, skip_4.shape)
        # print(e1_out.shape, e2_out.shape, e3_out.shape, e4_out.shape)

        d1_out = self.D1(e4_out, skip_1, anchors)
        # del e4_out, skip_1
        d2_out = self.D2(d1_out, skip_2, anchors)
        # del d1_out, skip_2
        d3_out = self.D3(d2_out, skip_3, anchors)
        # del d2_out, skip_3
        d4_out = self.D4(d3_out, skip_4, anchors)
        # del d3_out, skip_4
        logits = self.segmentation_head(d4_out)
        out = F.interpolate(logits, (D, H, W), mode="nearest")
        # out = self.final_upsample(logits)
        return out

    # def forward()

    # def debug_forward(self, X):
    #     embedding = self.embedding_stem(X)
    #     anchors = self.anchor_detector(embedding)
    #     T, f0 = self.lspm(embedding)
    #     # print(T, f0, anchors)
    #     e1_out, E1 = self.E1(f0, anchors, T)
    #     # print(e1_out.shape)
    #     e2_out, E2 = self.E2(e1_out, anchors, T)
    #     del e1_out
    #     e3_out, E3 = self.E3(e2_out, anchors, T)
    #     del e2_out
    #     e4_out, E4 = self.E4(e3_out, anchors, T)
    #     del e3_out
    #     skip_1, skip_2, skip_3, skip_4 = self.skip_fusion(
    #         E1, E2, E3, E4, input_shape=X.shape[2:]
    #     )
    #     del E1, E2, E3, E4
    #     # print(E1.shape, E2.shape, E3.shape, E4.shape)
    #     # print(skip_1.shape, skip_2.shape, skip_3.shape, skip_4.shape)
    #     # print(e1_out.shape, e2_out.shape, e3_out.shape, e4_out.shape)

    #     d1_out = self.D1(e4_out, skip_1, anchors)
    #     del e4_out, skip_1
    #     d2_out = self.D2(d1_out, skip_2, anchors)
    #     del d1_out, skip_2
    #     d3_out = self.D3(d2_out, skip_3, anchors)
    #     del d2_out, skip_3
    #     d4_out = self.D4(d3_out, skip_4, anchors)
    #     del d3_out, skip_4
    #     logits = self.segmentation_head(d4_out)

    #     # out = self.final_upsample(logits)
    #     out = F.interpolate(logits, (256, 256, 256), mode="trilinear")

    #     return out
