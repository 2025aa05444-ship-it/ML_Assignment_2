"""
Script to convert README.md to PDF
Requires: pip install markdown2pdf
If markdown2pdf doesn't work, falls back to pypandoc or reportlab approach
"""

import os
import sys
from pathlib import Path

def generate_pdf_with_markdown2pdf():
    """Generate PDF using markdown2pdf library"""
    try:
        from markdown2pdf.converter import convert
        
        input_file = "README.md"
        output_file = "Telecom_Churn_Prediction_Report.pdf"
        
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found!")
            return False
        
        print(f"Converting {input_file} to PDF...")
        convert(input_file, output_file)
        print(f"✅ PDF created successfully: {output_file}")
        return True
    except ImportError:
        print("markdown2pdf not installed. Trying alternative method...")
        return False

def generate_pdf_with_pypandoc():
    """Generate PDF using pypandoc (requires pandoc)"""
    try:
        import pypandoc
        
        input_file = "README.md"
        output_file = "Telecom_Churn_Prediction_Report.pdf"
        
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found!")
            return False
        
        print(f"Converting {input_file} to PDF using pypandoc...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        pdf_content = pypandoc.convert_text(
            readme_content,
            'pdf',
            format='md',
            outputfile=output_file,
            extra_args=['--pdf-engine=xelatex']
        )
        
        print(f"✅ PDF created successfully: {output_file}")
        return True
    except ImportError:
        print("pypandoc not installed. Trying alternative method...")
        return False
    except Exception as e:
        print(f"Error with pypandoc: {e}")
        return False

def generate_pdf_with_reportlab():
    """Generate PDF using reportlab library"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.lib import colors
        import re
        
        input_file = "README.md"
        output_file = "Telecom_Churn_Prediction_Report.pdf"
        
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found!")
            return False
        
        print(f"Converting {input_file} to PDF using reportlab...")
        
        # Read markdown file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create PDF
        doc = SimpleDocTemplate(output_file, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Add custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2e5c8a'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        
        # Parse markdown and add to story
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Title
            if line.startswith('# ') and not line.startswith('##'):
                title = line.replace('# ', '')
                story.append(Paragraph(title, title_style))
                story.append(Spacer(1, 0.3*inch))
            
            # Headings
            elif line.startswith('## '):
                heading = line.replace('## ', '')
                story.append(Paragraph(heading, heading_style))
            elif line.startswith('### '):
                heading = line.replace('### ', '')
                story.append(Paragraph(heading, styles['Heading3']))
            
            # Skip horizontal rules
            elif line.startswith('---'):
                story.append(Spacer(1, 0.15*inch))
            
            # Tables
            elif '|' in line and i + 2 < len(lines) and '|' in lines[i + 1]:
                table_rows = []
                j = i
                while j < len(lines) and '|' in lines[j]:
                    row = [cell.strip() for cell in lines[j].split('|')[1:-1]]
                    table_rows.append(row)
                    j += 1
                
                if len(table_rows) > 2:
                    table = Table(table_rows)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c8a')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 0.2*inch))
                i = j - 1
            
            # Code blocks
            elif line.startswith('```'):
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                code_text = '\n'.join(code_lines)
                story.append(Paragraph('<font face="Courier" size="9"><b>Code:</b></font>', styles['Normal']))
                story.append(Paragraph(f'<font face="Courier" size="8">{code_text}</font>', styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            # Regular paragraphs
            elif line and not line.startswith('#') and not line.startswith('-') and not line.startswith('|'):
                story.append(Paragraph(line, normal_style))
            
            # Bullet points
            elif line.startswith('-'):
                bullet = line.replace('-', '•', 1)
                story.append(Paragraph(bullet, normal_style))
            
            else:
                story.append(Spacer(1, 0.05*inch))
            
            i += 1
        
        # Add page break and footer
        doc.build(story)
        print(f"✅ PDF created successfully: {output_file}")
        return True
    except ImportError:
        print("reportlab not installed. Please install required package.")
        return False
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("README.md to PDF Converter")
    print("=" * 60)
    
    # Try different methods in order of preference
    if not generate_pdf_with_markdown2pdf():
        if not generate_pdf_with_pypandoc():
            if not generate_pdf_with_reportlab():
                print("\n❌ Failed to generate PDF. Please install one of the following:")
                print("   Option 1: pip install markdown2pdf")
                print("   Option 2: pip install pypandoc (requires pandoc)")
                print("   Option 3: pip install reportlab")
                sys.exit(1)
    
    print("\n" + "=" * 60)
    print("PDF generation completed!")
    print("=" * 60)
