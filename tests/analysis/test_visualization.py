"""Tests for visualization generation"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import numpy as np
from src.analysis.visualization import VisualizationGenerator


@pytest.fixture
def viz_generator(tmp_path):
    """Create VisualizationGenerator with temporary output directory"""
    return VisualizationGenerator(output_dir=str(tmp_path / "viz"))


@pytest.fixture
def sample_similarities():
    """Sample similarity scores for testing"""
    return [0.7, 0.8, 0.75, 0.85, 0.9, 0.72, 0.88, 0.76, 0.82, 0.79]


@pytest.fixture
def sample_technique_results():
    """Sample technique comparison data"""
    return {
        "baseline": [0.6, 0.65, 0.7, 0.68, 0.72],
        "standard": [0.75, 0.78, 0.8, 0.76, 0.79],
        "few_shot": [0.85, 0.88, 0.87, 0.9, 0.86],
        "chain_of_thought": [0.82, 0.84, 0.83, 0.86, 0.85]
    }


class TestVisualizationGenerator:
    """Test VisualizationGenerator class"""

    def test_initialization_creates_output_directory(self, tmp_path):
        """Test that output directory is created on initialization"""
        output_dir = tmp_path / "custom_viz"
        assert not output_dir.exists()

        viz = VisualizationGenerator(output_dir=str(output_dir))

        assert output_dir.exists()
        assert viz.output_dir == output_dir

    def test_initialization_with_existing_directory(self, tmp_path):
        """Test initialization when directory already exists"""
        output_dir = tmp_path / "existing"
        output_dir.mkdir()

        viz = VisualizationGenerator(output_dir=str(output_dir))

        assert output_dir.exists()
        assert viz.output_dir == output_dir

    @patch('matplotlib.pyplot.rcParams', {})
    @patch('seaborn.set_style')
    def test_initialization_sets_matplotlib_style(self, mock_seaborn_style, tmp_path):
        """Test that matplotlib style is configured on init"""
        viz = VisualizationGenerator(output_dir=str(tmp_path))

        # Verify seaborn style was set
        mock_seaborn_style.assert_called_once_with("whitegrid")

    def test_default_output_directory(self):
        """Test default output directory path"""
        viz = VisualizationGenerator()
        assert viz.output_dir == Path("results/visualizations")


class TestPlotHistogram:
    """Test histogram generation"""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.hist')
    @patch('matplotlib.pyplot.axvline')
    def test_plot_histogram_basic(
        self,
        mock_axvline,
        mock_hist,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator,
        sample_similarities
    ):
        """Test basic histogram creation"""
        viz_generator.plot_histogram(
            similarities=sample_similarities,
            title="Test Histogram",
            filename="test_hist.png"
        )

        # Verify plot functions called
        assert mock_figure.called
        mock_hist.assert_called_once()
        mock_axvline.assert_called_once()
        mock_close.assert_called_once()

        # Verify histogram data
        call_args = mock_hist.call_args
        np.testing.assert_array_equal(call_args[0][0], sample_similarities)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    def test_plot_histogram_saves_to_correct_path(
        self,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test that histogram is saved to correct file path"""
        similarities = [0.7, 0.8, 0.9]
        filename = "output.png"

        viz_generator.plot_histogram(
            similarities=similarities,
            title="Test",
            filename=filename
        )

        expected_path = viz_generator.output_dir / filename
        mock_savefig.assert_called_once()
        actual_path = mock_savefig.call_args[0][0]
        assert actual_path == expected_path

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.axvline')
    def test_plot_histogram_includes_mean_line(
        self,
        mock_axvline,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test that mean line is added to histogram"""
        similarities = [0.6, 0.7, 0.8, 0.9]
        expected_mean = np.mean(similarities)

        viz_generator.plot_histogram(
            similarities=similarities,
            title="Test",
            filename="test.png"
        )

        # Verify axvline called with mean
        mock_axvline.assert_called_once()
        actual_mean = mock_axvline.call_args[0][0]
        assert actual_mean == pytest.approx(expected_mean)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.hist')
    def test_plot_histogram_with_empty_list(
        self,
        mock_hist,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test histogram with empty similarity list handles gracefully"""
        # Matplotlib handles empty lists without error
        viz_generator.plot_histogram(
            similarities=[],
            title="Empty",
            filename="empty.png"
        )

        # Should still create plot
        assert mock_figure.called
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    def test_plot_histogram_with_single_value(
        self,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test histogram with single value"""
        viz_generator.plot_histogram(
            similarities=[0.75],
            title="Single Value",
            filename="single.png"
        )

        mock_savefig.assert_called_once()


class TestPlotComparisonBars:
    """Test bar chart generation"""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.bar')
    def test_plot_comparison_bars_basic(
        self,
        mock_bar,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator,
        sample_technique_results
    ):
        """Test basic bar chart creation"""
        viz_generator.plot_comparison_bars(
            technique_results=sample_technique_results,
            title="Technique Comparison",
            filename="comparison.png"
        )

        # Verify plot functions called
        assert mock_figure.called
        mock_bar.assert_called_once()
        mock_close.assert_called_once()
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.bar')
    def test_plot_comparison_bars_correct_data(
        self,
        mock_bar,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator,
        sample_technique_results
    ):
        """Test that bar chart uses correct means and stds"""
        viz_generator.plot_comparison_bars(
            technique_results=sample_technique_results,
            title="Test",
            filename="test.png"
        )

        # Extract call arguments
        call_args = mock_bar.call_args
        techniques = call_args[0][0]
        means = call_args[0][1]

        # Verify techniques
        assert techniques == list(sample_technique_results.keys())

        # Verify means calculated correctly
        expected_means = [np.mean(scores) for scores in sample_technique_results.values()]
        np.testing.assert_array_almost_equal(means, expected_means)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.bar')
    def test_plot_comparison_bars_includes_error_bars(
        self,
        mock_bar,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test that error bars (yerr) are included"""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}

        viz_generator.plot_comparison_bars(
            technique_results=data,
            title="Test",
            filename="test.png"
        )

        # Verify yerr parameter passed
        call_kwargs = mock_bar.call_args[1]
        assert 'yerr' in call_kwargs
        assert call_kwargs['yerr'] is not None

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    def test_plot_comparison_bars_saves_to_correct_path(
        self,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test that bar chart is saved to correct file"""
        data = {"baseline": [0.6, 0.7]}
        filename = "bars.png"

        viz_generator.plot_comparison_bars(
            technique_results=data,
            title="Test",
            filename=filename
        )

        expected_path = viz_generator.output_dir / filename
        actual_path = mock_savefig.call_args[0][0]
        assert actual_path == expected_path

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.bar')
    @patch('matplotlib.pyplot.text')
    def test_plot_comparison_bars_adds_value_labels(
        self,
        mock_text,
        mock_bar,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test that value labels are added to bars"""
        data = {"A": [0.7, 0.8], "B": [0.85, 0.9]}

        # Mock the bar objects
        mock_bar_obj1 = Mock()
        mock_bar_obj1.get_height.return_value = 0.75
        mock_bar_obj1.get_x.return_value = 0
        mock_bar_obj1.get_width.return_value = 1

        mock_bar_obj2 = Mock()
        mock_bar_obj2.get_height.return_value = 0.875
        mock_bar_obj2.get_x.return_value = 1
        mock_bar_obj2.get_width.return_value = 1

        mock_bar.return_value = [mock_bar_obj1, mock_bar_obj2]

        viz_generator.plot_comparison_bars(
            technique_results=data,
            title="Test",
            filename="test.png"
        )

        # Verify text labels added for each bar
        assert mock_text.call_count == 2


class TestPlotBoxPlots:
    """Test box plot generation"""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.boxplot')
    def test_plot_box_plots_basic(
        self,
        mock_boxplot,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator,
        sample_technique_results
    ):
        """Test basic box plot creation"""
        viz_generator.plot_box_plots(
            technique_results=sample_technique_results,
            title="Distribution Comparison",
            filename="boxplot.png"
        )

        # Verify plot functions called
        assert mock_figure.called
        mock_boxplot.assert_called_once()
        mock_close.assert_called_once()
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.boxplot')
    def test_plot_box_plots_correct_data(
        self,
        mock_boxplot,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator,
        sample_technique_results
    ):
        """Test that box plot uses correct data"""
        viz_generator.plot_box_plots(
            technique_results=sample_technique_results,
            title="Test",
            filename="test.png"
        )

        # Extract call arguments
        call_args = mock_boxplot.call_args
        data = call_args[0][0]
        call_kwargs = call_args[1]
        labels = call_kwargs['labels']

        # Verify data and labels
        assert data == list(sample_technique_results.values())
        assert labels == list(sample_technique_results.keys())

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.boxplot')
    def test_plot_box_plots_saves_to_correct_path(
        self,
        mock_boxplot,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test that box plot is saved to correct file"""
        data = {"A": [0.6, 0.7, 0.8]}
        filename = "boxes.png"

        # Mock boxplot return value
        mock_boxplot.return_value = {'boxes': []}

        viz_generator.plot_box_plots(
            technique_results=data,
            title="Test",
            filename=filename
        )

        expected_path = viz_generator.output_dir / filename
        actual_path = mock_savefig.call_args[0][0]
        assert actual_path == expected_path

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.boxplot')
    def test_plot_box_plots_colors_boxes(
        self,
        mock_boxplot,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test that boxes are colored"""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}

        # Mock boxplot return with box patches
        mock_patch1 = Mock()
        mock_patch2 = Mock()
        mock_boxplot.return_value = {
            'boxes': [mock_patch1, mock_patch2]
        }

        viz_generator.plot_box_plots(
            technique_results=data,
            title="Test",
            filename="test.png"
        )

        # Verify patches were colored
        mock_patch1.set_facecolor.assert_called_once_with('lightblue')
        mock_patch2.set_facecolor.assert_called_once_with('lightblue')


class TestVisualizationIntegration:
    """Integration tests for visualization pipeline"""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    def test_multiple_plots_in_sequence(
        self,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator,
        sample_similarities,
        sample_technique_results
    ):
        """Test creating multiple plots in sequence"""
        # Create histogram
        viz_generator.plot_histogram(
            similarities=sample_similarities,
            title="Histogram",
            filename="hist.png"
        )

        # Create bar chart
        viz_generator.plot_comparison_bars(
            technique_results=sample_technique_results,
            title="Bars",
            filename="bars.png"
        )

        # Create box plot
        with patch('matplotlib.pyplot.boxplot', return_value={'boxes': []}):
            viz_generator.plot_box_plots(
                technique_results=sample_technique_results,
                title="Boxes",
                filename="boxes.png"
            )

        # Verify all three plots were saved
        assert mock_savefig.call_count == 3

        # Verify all plots were closed
        assert mock_close.call_count == 3

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    def test_all_plots_save_with_high_dpi(
        self,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test that all plots are saved with publication quality (300 DPI)"""
        # Histogram
        viz_generator.plot_histogram(
            similarities=[0.7, 0.8],
            title="Test",
            filename="test1.png"
        )

        # Bar chart
        viz_generator.plot_comparison_bars(
            technique_results={"A": [0.7]},
            title="Test",
            filename="test2.png"
        )

        # Box plot
        with patch('matplotlib.pyplot.boxplot', return_value={'boxes': []}):
            viz_generator.plot_box_plots(
                technique_results={"A": [0.7]},
                title="Test",
                filename="test3.png"
            )

        # Verify all savefig calls include dpi=300
        for call_obj in mock_savefig.call_args_list:
            assert call_obj[1]['dpi'] == 300


class TestEdgeCases:
    """Test edge cases and error handling"""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    def test_plot_with_identical_values(
        self,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test plotting when all values are identical (std=0)"""
        identical_data = {"A": [0.8, 0.8, 0.8, 0.8]}

        # Should not raise error even with zero variance
        viz_generator.plot_comparison_bars(
            technique_results=identical_data,
            title="Test",
            filename="test.png"
        )

        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.boxplot')
    def test_plot_with_extreme_values(
        self,
        mock_boxplot,
        mock_figure,
        mock_close,
        mock_savefig,
        viz_generator
    ):
        """Test plotting with extreme outliers"""
        data = {"A": [0.0, 0.5, 1.0]}  # Full range

        mock_boxplot.return_value = {'boxes': []}

        viz_generator.plot_box_plots(
            technique_results=data,
            title="Test",
            filename="test.png"
        )

        mock_savefig.assert_called_once()

    def test_output_directory_permission_error(self, tmp_path):
        """Test handling when output directory cannot be created"""
        # Create a file where directory should be
        conflicting_path = tmp_path / "conflict"
        conflicting_path.write_text("blocking file")

        # Attempting to create directory should raise error
        with pytest.raises(Exception):
            # Will fail because conflict is a file, not a directory
            viz = VisualizationGenerator(output_dir=str(conflicting_path / "subdir"))
            viz.output_dir.mkdir(parents=True, exist_ok=False)
