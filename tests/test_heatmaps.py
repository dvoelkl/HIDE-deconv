"""
Tests for heatmap visualization helpers.
"""

import pandas as pd

import hide_deconv.visualization.heatmaps as heatmaps


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class MockHeatmapAxis:
    """
    Capture tick parameter settings for the heatmap axis.
    """

    def __init__(self):
        self.calls = []

    def tick_params(self, **kwargs):
        self.calls.append(kwargs)


class MockClusterMap:
    """
    Minimal clustermap result used to capture savefig calls.
    """

    def __init__(self):
        self.ax_heatmap = MockHeatmapAxis()
        self.saved_path = None

    def savefig(self, out_path):
        self.saved_path = out_path


class TestPlotGeneMap:
    """
    Tests for plot_genemap.
    """

    def test_plot_genemap_uses_minimum_figure_size_for_small_inputs(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that small inputs still use the minimum plot size and format labels.
        """

        data = pd.DataFrame(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [2, 3, 4],
            ],
            index=["gene_1", "gene_2", "gene_3", "gene_4"],
            columns=["ct_1", "ct_2", "ct_3"],
        )

        captured = {}
        mock_clustermap = MockClusterMap()

        def capture_clustermap(*args, **kwargs):
            captured["data_shape"] = args[0].shape
            captured["figsize"] = kwargs["figsize"]
            return mock_clustermap

        monkeypatch.setattr(heatmaps.sns, "clustermap", capture_clustermap)
        monkeypatch.setattr(heatmaps.plt, "title", lambda *args, **kwargs: None)

        out_path = tmp_path / "markermap.png"

        heatmaps.plot_genemap(data, ["gene_1", "gene_2", "gene_3"], "Title", out_path)

        assert captured["data_shape"] == (3, 3)
        assert captured["figsize"] == (25, 6)
        assert mock_clustermap.saved_path == out_path
        assert mock_clustermap.ax_heatmap.calls == [
            {"axis": "x", "labelrotation": 90, "labelsize": 8},
            {"axis": "y", "labelsize": 8},
        ]

    def test_plot_genemap_scales_figure_with_more_labels(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that larger inputs increase the plot size to reduce label overlap.
        """

        data = pd.DataFrame(
            [[i + j for j in range(40)] for i in range(120)],
            index=[f"gene_{i}" for i in range(1, 121)],
            columns=[f"ct_{i}" for i in range(1, 41)],
        )

        captured = {}
        mock_clustermap = MockClusterMap()

        def capture_clustermap(*args, **kwargs):
            captured["figsize"] = kwargs["figsize"]
            return mock_clustermap

        monkeypatch.setattr(heatmaps.sns, "clustermap", capture_clustermap)
        monkeypatch.setattr(heatmaps.plt, "title", lambda *args, **kwargs: None)

        out_path = tmp_path / "markermap_large.png"

        heatmaps.plot_genemap(
            data,
            [f"gene_{i}" for i in range(1, 121)],
            "Title",
            out_path,
        )

        assert captured["figsize"] == (54.0, 14.0)
        assert mock_clustermap.saved_path == out_path
        assert len(mock_clustermap.ax_heatmap.calls) == 2
