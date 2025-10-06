from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Scientific Research Report", ln=True, align="C")
        self.ln(10)

    def add_section(self, title: str, content: str):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, title.title(), ln=True)
        self.set_font("Helvetica", "", 12)
        self.multi_cell(0, 8, content)
        self.ln(5)


def generate_pdf_with_images(data: dict, images: list[str], output_path: str):
    pdf = PDF()
    pdf.add_page()

    # Title
    title = data.get("title", "Scientific Research Report")
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, title, ln=True, align="C")
    pdf.ln(10)

    # Add text sections
    for key, value in data.items():
        if key == "title":
            continue
        pdf.add_section(key.replace("_", " "), str(value))

    # Add images
    for img_path in images:
        pdf.image(img_path, w=170)
        pdf.ln(10)

    pdf.output(output_path)
    return output_path
