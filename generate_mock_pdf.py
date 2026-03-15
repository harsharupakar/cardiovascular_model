from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_mock_echo_report(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Title & Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "ECHOCARDIOGRAPHY REPORT")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, "Patient Name: Jane Doe")
    c.drawString(50, height - 100, "Date of Birth: 05/15/1995 (Age: 30)")
    c.drawString(50, height - 120, "Date of Exam: March 15, 2026")
    
    c.line(50, height - 130, width - 50, height - 130)
    
    # Measurements Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 160, "Key Measurements & Findings:")
    
    c.setFont("Helvetica", 12)
    # Give it moderate-risk values based on our trained rules
    # LVEF = 42 (Low, indicates HF/Cardiomyopathy)
    # LVEDD = 58 (High, indicates Cardiomyopathy)
    c.drawString(70, height - 190, "- Left Ventricular Ejection Fraction (LVEF): 42%")
    c.drawString(70, height - 210, "- Left Ventricular End-Diastolic Dimension (LVEDD): 58 mm")
    
    # Valve & Motion
    # Wall Motion = Hypokinesia (indicates CAD)
    # Mitral Regurg = Moderate (indicates HF)
    c.drawString(70, height - 230, "- Wall Motion Abnormalities: Hypokinesia observed in the anterior wall.")
    c.drawString(70, height - 250, "- Mitral Valve: Moderate Mitral Regurgitation noted.")
    
    c.drawString(70, height - 270, "- E/A Ratio: 0.8 (indicates Grade I Diastolic Dysfunction)")
    c.drawString(70, height - 290, "- PASP: 42 mmHg (indicates mild pulmonary hypertension)")
    c.drawString(70, height - 310, "- Left Atrial Volume Index (LAVI): 45 mL/m2 (severe enlargement)")
    c.drawString(70, height - 330, "- Aortic Valve Area: 1.2 cm2 (moderate stenosis)")
    
    # Conclusion
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 370, "Conclusion / Impression:")
    
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 400, "1. Mildly reduced left ventricular systolic function.")
    c.drawString(70, height - 420, "2. Left ventricular enlargement.")
    c.drawString(70, height - 440, "3. Moderate mitral regurgitation and moderate aortic stenosis.")
    c.drawString(70, height - 460, "4. Evidence of diastolic dysfunction and pulmonary hypertension.")
    
    c.save()

if __name__ == "__main__":
    create_mock_echo_report("mock_echo_report_v2.pdf")
    print("PDF generated successfully.")
