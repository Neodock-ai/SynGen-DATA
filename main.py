"""
main.py

An industry-level Flet web application for synthetic data generation.
Users can upload a CSV dataset, specify the desired number of synthetic rows,
and then generate and download a synthetic dataset with similar structure and distribution.
"""

import flet as ft
import pandas as pd
import io
import logging
from synthetic_generator import generate_synthetic_data  # Your data generation logic

# Configure logging for production use.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def main(page: ft.Page):
    # Set up page metadata and styling
    page.title = "Synthetic Data Generator"
    page.padding = 20
    page.vertical_alignment = "start"

    # Global variables to hold the original and synthetic data
    original_data = None
    synthetic_data_df = None

    # Header for the app
    header = ft.Text("Synthetic Data Generator", style="headlineMedium", color=ft.colors.BLUE)

    # Create a FilePicker control to allow file uploads
    file_picker = ft.FilePicker(on_result=lambda e: file_picker_result(e))
    page.overlay.append(file_picker)

    # TextField for user to input desired number of synthetic rows
    rows_input = ft.TextField(
        label="Number of Synthetic Rows", 
        value="100", 
        width=300,
        tooltip="Enter a positive integer specifying how many rows of synthetic data you need."
    )

    # Button to trigger file upload; accepts CSV files only
    upload_button = ft.ElevatedButton(
        text="Upload CSV Dataset",
        icon=ft.icons.UPLOAD_FILE,
        on_click=lambda _: file_picker.pick_files(allow_multiple=False, file_type="csv")
    )

    # Button to generate synthetic data; disabled until a dataset is uploaded
    generate_button = ft.ElevatedButton(
        text="Generate Synthetic Data",
        icon=ft.icons.DATA_SAVER_OFF,
        on_click=lambda _: on_generate(),
        disabled=True
    )

    # A FilePickerUploadButton for downloading the generated synthetic CSV.
    download_button = ft.FilePickerUploadButton(
        file_name="synthetic_data.csv",
        file_bytes=b"",
        visible=False,
        text="Download Synthetic Data",
        icon=ft.icons.DOWNLOAD
    )

    # Text control to show status messages to the user
    status_text = ft.Text("", color=ft.colors.GREEN)

    # Column to hold dynamic output (e.g. progress bar, preview)
    output_container = ft.Column(controls=[], spacing=10)

    def file_picker_result(e: ft.FilePickerResultEvent):
        nonlocal original_data
        # Called when the file picker returns a result
        if e.files and len(e.files) > 0:
            try:
                file = e.files[0]
                # Read file bytes and decode as UTF-8 text
                content = file.read_bytes()
                data_io = io.StringIO(content.decode("utf-8"))
                # Load CSV into a pandas DataFrame
                original_data = pd.read_csv(data_io)
                logging.info("Uploaded dataset with shape: %s", original_data.shape)
                status_text.value = f"Dataset uploaded successfully with shape {original_data.shape}."
                status_text.color = ft.colors.GREEN
                generate_button.disabled = False  # Enable synthetic generation
            except Exception as ex:
                logging.error("Error processing uploaded file: %s", ex)
                status_text.value = f"Error reading file: {ex}"
                status_text.color = ft.colors.RED
                generate_button.disabled = True
            finally:
                page.update()

    def on_generate():
        nonlocal synthetic_data_df
        if original_data is None:
            status_text.value = "Please upload a dataset first."
            status_text.color = ft.colors.RED
            page.update()
            return

        # Validate and parse the desired number of synthetic rows
        try:
            num_rows = int(rows_input.value)
            if num_rows <= 0:
                raise ValueError("Number of rows must be a positive integer.")
        except ValueError as ve:
            status_text.value = "Invalid number of rows. Please enter a positive integer."
            status_text.color = ft.colors.RED
            page.update()
            return

        # Provide user feedback that generation is in progress
        status_text.value = "Generating synthetic data. Please wait..."
        status_text.color = ft.colors.BLUE
        page.update()

        # Clear previous output and display a progress indicator
        output_container.controls.clear()
        progress_bar = ft.ProgressBar(width=300, visible=True)
        output_container.controls.append(progress_bar)
        page.update()

        try:
            # Call your synthetic data generation function
            synthetic_data_df = generate_synthetic_data(original_data, num_rows)
            logging.info("Synthetic data generated with shape: %s", synthetic_data_df.shape)
            status_text.value = f"Synthetic data generated with shape {synthetic_data_df.shape}."
            status_text.color = ft.colors.GREEN

            # Remove the progress bar and display a preview of the synthetic data
            output_container.controls.clear()
            preview_label = ft.Text("Preview of Synthetic Data (first 5 rows):", weight="bold")
            # Convert the first 5 rows of the DataFrame to a string (monospaced for readability)
            preview_str = synthetic_data_df.head(5).to_string(index=False)
            preview_text = ft.Text(preview_str, font_family="monospace", size=12)
            output_container.controls.extend([preview_label, preview_text])

            # Prepare the download button: convert DataFrame to CSV bytes
            csv_bytes = synthetic_data_df.to_csv(index=False).encode("utf-8")
            download_button.file_bytes = csv_bytes
            download_button.visible = True
        except Exception as ex:
            logging.error("Error during synthetic data generation: %s", ex)
            status_text.value = f"Error generating synthetic data: {ex}"
            status_text.color = ft.colors.RED
            output_container.controls.clear()
        finally:
            page.update()

    # Layout the app controls in a responsive design
    page.add(
        header,
        ft.Row(
            controls=[upload_button, rows_input, generate_button],
            alignment="start",
            spacing=20
        ),
        status_text,
        output_container,
        download_button
    )

# Launch the Flet app in the web browser.
ft.app(target=main, view=ft.WEB_BROWSER)
