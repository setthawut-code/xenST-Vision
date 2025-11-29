"""
Web Interface for Spatial Transcriptomics Prediction
Professional Academic Theme - English Only
"""

import gradio as gr
import numpy as np
import torch
import cv2
from pathlib import Path
import sys
import tempfile
import pandas as pd
from typing import Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference import SpatialInference
from utils import load_config, plot_interactive_heatmap


class SpatialTranscriptomicsApp:
    """Gradio app wrapper"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.inference_engine = None
        self.latest_predictions = None
        self.latest_coordinates = None
        self.gene_names = []
        
        # Find available models
        self.available_models = self._find_models()
    
    def _find_models(self):
        """Find available model checkpoints"""
        checkpoint_dir = Path(self.config['paths']['checkpoints_dir'])
        if not checkpoint_dir.exists():
            return []
        
        models = []
        for ckpt in checkpoint_dir.glob("*.pth"):
            models.append(str(ckpt))
        
        return models if models else ["No models available - please train first"]
    
    def load_model(self, model_path: str):
        """Load inference model"""
        try:
            if not Path(model_path).exists():
                return "Model not found"
            
            self.inference_engine = SpatialInference(
                model_path, 
                self.config,
                device=self.config['inference']['device']
            )
            self.gene_names = self.inference_engine.gene_names
            
            return f"Model loaded successfully\\n{len(self.gene_names)} genes"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def predict(
        self,
        image: np.ndarray,
        model_path: str,
        selected_gene: str,
        progress=gr.Progress()
    ) -> Tuple[str, str, pd.DataFrame, str]:
        """Run prediction on uploaded image"""
        if image is None:
            return "Please upload an image", None, None, None
        
        if not self.inference_engine or not Path(model_path).exists():
            status = self.load_model(model_path)
            if "Error" in status:
                return status, None, None, None
        
        try:
            progress(0.1, desc="Processing image...")
            
            # Save temp image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            progress(0.3, desc="Extracting patches...")
            
            # Create temp output dir
            output_dir = tempfile.mkdtemp()
            
            progress(0.5, desc="Predicting gene expression...")
            
            # Run prediction
            results = self.inference_engine.predict_image(
                tmp_path,
                output_dir,
                patch_size=self.config['data']['patch_size']
            )
            
            progress(0.8, desc="Creating visualizations...")
            
            # Store results
            self.latest_predictions = results['predictions']
            self.latest_coordinates = np.array(results['coordinates'])
            
            # Create predictions DataFrame
            pred_df = pd.DataFrame(
                self.latest_predictions,
                columns=self.gene_names
            )
            pred_df.insert(0, 'x', self.latest_coordinates[:, 0])
            pred_df.insert(1, 'y', self.latest_coordinates[:, 1])
            
            # Get selected gene index
            if selected_gene in self.gene_names:
                gene_idx = self.gene_names.index(selected_gene)
            else:
                gene_idx = 0
            
            # Create interactive plot
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as tmp_html:
                html_path = tmp_html.name
                plot_interactive_heatmap(
                    self.latest_predictions[:, gene_idx],
                    self.latest_coordinates,
                    selected_gene,
                    html_path
                )
                
                with open(html_path, 'r') as f:
                    html_content = f.read()
            
            # CSV for download
            csv_path = Path(output_dir) / "predictions.csv"
            pred_df.to_csv(csv_path, index=False)
            
            progress(1.0, desc="Complete!")
            
            status_msg = f"""
            **Prediction Complete**
            
            - Predicted spots: {len(self.latest_predictions)}
            - Genes: {len(self.gene_names)}
            - Expected correlation: 0.3-0.5
            """
            
            return status_msg, html_content, pred_df, str(csv_path)
            
        except Exception as e:
            return f"Error: {str(e)}", None, None, None


def create_interface():
    """Create Gradio interface"""
    app = SpatialTranscriptomicsApp()
    
    # Professional CSS - ALL DARK TEXT ON WHITE
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        * {
            color: #1e293b !important;
        }
        
        body {
            font-family: 'Inter', sans-serif !important;
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2) !important;
        }
        
        .gradio-container {
            max-width: 1400px !important;
            margin: 0 auto !important;
            background: white !important;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1) !important;
            border-radius: 12px !important;
            padding: 30px !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #1e3a8a !important;
            font-weight: 700 !important;
        }
        
        h1 {
            border-bottom: 3px solid #3b82f6 !important;
            padding-bottom: 15px !important;
        }
        
        h3 {
            color: #475569 !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            border-left: 4px solid #3b82f6 !important;
            padding-left: 12px !important;
        }
        
        p, span, div, label, strong, em, li {
            color: #334155 !important;
        }
        
        label {
            font-weight: 600 !important;
            text-transform: uppercase !important;
            font-size: 0.9rem !important;
        }
        
        input, textarea, select {
            background: #f8fafc !important;
            border: 2px solid #e2e8f0 !important;
            color: #1e293b !important;
            border-radius: 8px !important;
        }
        
        button {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            border: none !important;
        }
        
        button:hover {
            background: linear-gradient(135deg, #2563eb, #1e40af) !important;
        }
        
        table {
            background: white !important;
            border: 1px solid #e2e8f0 !important;
        }
        
        table th {
            background: #1e40af !important;
            color: white !important;
            font-weight: 600 !important;
        }
        
        table td {
            color: #1e293b !important;
            background: white !important;
        }
        
        .info-box {
            background: #eff6ff !important;
            border: 1px solid #93c5fd !important;
            border-radius: 8px !important;
            padding: 16px !important;
        }
        
        .info-box * {
            color: #1e40af !important;
        }
        
        .info-box strong {
            color: #1e3a8a !important;
        }
    </style>
    """
    
    with gr.Blocks(title="Spatial Transcriptomics Platform") as demo:
        gr.HTML(css)
        
        gr.Markdown("""
        # Spatial Transcriptomics Prediction Platform
        ### Deep Learning-Based Gene Expression Prediction from H&E Images
        """)
        
        gr.Markdown("""
        <div class="info-box">
        <strong>Research Application:</strong> 
        Deep learning prediction of spatial gene expression from H&E histology images, 
        trained with Xenium spatial transcriptomics data.
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### DATA INPUT")
                
                image_input = gr.Image(
                    label="H&E Histology Image",
                    type="numpy"
                )
                
                gr.Markdown("### CONFIGURATION")
                
                model_dropdown = gr.Dropdown(
                    choices=app.available_models,
                    label="Model Selection",
                    value=app.available_models[0] if app.available_models else None
                )
                
                gene_dropdown = gr.Dropdown(
                    choices=app.gene_names if app.gene_names else ["Load model first"],
                    label="Target Gene",
                    value=app.gene_names[0] if app.gene_names else None
                )
                
                predict_btn = gr.Button(
                    "RUN ANALYSIS",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                **INSTRUCTIONS:**
                1. Upload H&E tissue image
                2. Select prediction model
                3. Choose target gene
                4. Run analysis
                5. Review results
                6. Export data
                """)
            
            with gr.Column(scale=3):
                gr.Markdown("### RESULTS")
                
                status_output = gr.Markdown(value="Awaiting data...")
                
                gr.Markdown("#### Spatial Expression Map")
                plot_output = gr.HTML(
                    value="<div style='padding: 40px; text-align: center; color: #1e293b; background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 8px;'>Results will appear here</div>"
                )
                
                with gr.Accordion("PREDICTIONS TABLE", open=False):
                    table_output = gr.Dataframe(label="Gene Expression Matrix")
                
                gr.Markdown("#### EXPORT")
                download_output = gr.File(label="Download CSV")
        
        gr.Markdown("""
        ---
        <div style='text-align: center;'>
        <strong>Spatial Transcriptomics Platform</strong> | ResNet50 Architecture | Xenium Data | Research Use
        </div>
        """)
        
        predict_btn.click(
            fn=app.predict,
            inputs=[image_input, model_dropdown, gene_dropdown],
            outputs=[status_output, plot_output, table_output, download_output]
        )
        
        def update_genes(model_path):
            if Path(model_path).exists():
                app.load_model(model_path)
                return gr.Dropdown(choices=app.gene_names, value=app.gene_names[0] if app.gene_names else None)
            return gr.Dropdown(choices=["Load model first"], value=None)
        
        model_dropdown.change(
            fn=update_genes,
            inputs=[model_dropdown],
            outputs=[gene_dropdown]
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    
    demo = create_interface()
    demo.launch(share=args.share, server_port=args.port, server_name="0.0.0.0")
