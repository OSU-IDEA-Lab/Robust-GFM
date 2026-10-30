"""ParitySlotGAT must reproduce the upstream DGL ``slotGAT`` with the same weights.

Runs on IMDB_var1_year (pooled-BoW features) on whichever device is available: the
upstream model is instantiated through the runner's own ``build_original_model`` (so it is
exactly the ``original`` arm), its weights are copied into ``ParitySlotGAT``, and both are
evaluated on the SAME edge list (upstream's raw graph + self loops, 7 edge types).  Logits
must agree to float tolerance (DGL's edge_softmax / scatter kernels are not bitwise), and so
must the gradients of the training loss.  A second check confirms that the six documented
departures are the ONLY differences: with every ablation switched on the parity model has
exactly the re-implementation's parameter count.
"""
from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

import run_IMDB_nc_conditional as R  # noqa: E402
from model.genuine_conditional_slotgat import GenuineConditionalSlotGAT, build_bucket_specs  # noqa: E402
from model.parity_slotgat import ABLATIONS, ParitySlotGAT  # noqa: E402


def _args() -> argparse.Namespace:
    return argparse.Namespace(num_heads=8, num_layers=2, edge_feats=64, hidden_dim=64, dropout_feat=0.5,
                              dropout_attn=0.2, slope=0.05, aggregator="SA", SAattDim=3, feats_type=0)


class ParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        R.VARIANT_NAMES[1] = "IMDB_var1_year"
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.bundle = R.load_datasets([1], strict=True, feature_mode="pooled_bow")[1]
        cls.features, cls.in_dims = R.tensorize_features(cls.bundle.features_raw, 0, cls.device)
        cls.labels = torch.as_tensor(cls.bundle.labels_np, dtype=torch.int64, device=cls.device)
        cls.train_idx = torch.as_tensor(cls.bundle.split_indices["train"], dtype=torch.int64, device=cls.device)

    def _pair(self, seed=0):
        torch.manual_seed(seed)
        upstream, up_fwd, meta = R.build_original_model(_args(), features_list=self.features, in_dims=self.in_dims,
                                                        dl=self.bundle.dl, device=self.device, data_recorder=R.make_data_recorder())
        graph, e_feat, num_et = R.build_original_graph(features_list=self.features, dl=self.bundle.dl, device=self.device)
        src, dst = graph.edges()
        parity = ParitySlotGAT(layout=self.bundle.physical_program.layout, in_dims=self.in_dims, num_classes=R.NUM_CLASSES,
                               hidden_dim=64, num_heads=8, num_layers=2, edge_feats=64, dropout_feat=0.5, dropout_attn=0.2,
                               slope=0.05, aggregator="SA", sa_attention_dim=3, num_edge_types=num_et).to(self.device)
        parity.load_from_upstream(upstream)
        n = int(graph.num_nodes())
        dbs, sbs = build_bucket_specs(dst, n), build_bucket_specs(src, n)
        par_fwd = lambda feats: parity.forward_edges(feats, src, dst, e_feat, dbs, sbs)
        return upstream, up_fwd, parity, par_fwd

    def test_parameter_counts(self):
        upstream, _, parity, _ = self._pair()
        n_up = sum(p.numel() for p in upstream.parameters()) - upstream.last_fc.numel()   # last_fc: unused (aggregator "last_fc" only)
        n_par = sum(p.numel() for p in parity.parameters())
        self.assertEqual(n_up, n_par, "parity model must have exactly upstream's parameters")
        # all ablations on == the re-implementation (up to the channel vocabulary of the edge table)
        genuine = GenuineConditionalSlotGAT(layout=self.bundle.physical_program.layout, in_dims=self.in_dims, num_classes=R.NUM_CLASSES,
                                            hidden_dim=64, num_heads=8, num_layers=2, edge_feats=64, dropout_feat=0.5,
                                            dropout_attn=0.2, slope=0.05, aggregator="SA", sa_attention_dim=3)
        ablated = ParitySlotGAT(layout=self.bundle.physical_program.layout, in_dims=self.in_dims, num_classes=R.NUM_CLASSES,
                                hidden_dim=64, num_heads=8, num_layers=2, edge_feats=64, dropout_feat=0.5, dropout_attn=0.2,
                                slope=0.05, aggregator="SA", sa_attention_dim=3, ablate=ABLATIONS)
        self.assertEqual(sum(p.numel() for p in genuine.parameters()), sum(p.numel() for p in ablated.parameters()))
        print(f"\n[params] upstream {n_up:,} = parity {n_par:,}; re-implementation {sum(p.numel() for p in genuine.parameters()):,}")

    def test_logits_and_gradients_match(self):
        upstream, up_fwd, parity, par_fwd = self._pair()
        upstream.eval(); parity.eval()
        with torch.no_grad():
            a, _ = up_fwd(self.features)
            b, _ = par_fwd(self.features)
        diff = (a - b).abs().max().item()
        agree = (a.argmax(1) == b.argmax(1)).float().mean().item()
        # the final logits are L2-normalised from an output layer of magnitude ~1e-3, so
        # float32 kernel-order noise is amplified ~1e3x; 1e-3 on unit vectors is numerical
        print(f"[eval logits] max |upstream - parity| = {diff:.2e}, argmax agreement {agree:.6f}")
        self.assertLess(diff, 2e-3); self.assertGreater(agree, 0.999)
        # training loss gradient (dropout off: eval mode keeps the comparison deterministic)
        for model, fwd in ((upstream, up_fwd), (parity, par_fwd)):
            model.zero_grad(); logits, _ = fwd(self.features)
            F.nll_loss(F.log_softmax(logits, 1)[self.train_idx], self.labels[self.train_idx]).backward()
        pairs = [(upstream.fc_list[0].weight, parity.input_projections[0].weight),
                 (upstream.gat_layers[1].fc, parity.layers[1].fc), (upstream.gat_layers[1].res_fc, parity.layers[1].res_fc),
                 (upstream.gat_layers[-1].attn_e, parity.layers[-1].attn_e), (upstream.macroLinear.weight, parity.macroLinear.weight)]
        for u, p in pairs:
            rel = (u.grad - p.grad).abs().max().item() / (u.grad.abs().max().item() + 1e-12)
            self.assertLess(rel, 1e-3, f"gradient mismatch {rel:.2e}")
        print(f"[gradients] max relative mismatch over 5 tensors < 1e-3")


if __name__ == "__main__":
    unittest.main()
