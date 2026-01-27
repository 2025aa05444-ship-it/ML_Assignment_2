"""
Professional PDF Generator for Assignment Submission
Generates a high-quality PDF with:
- Professional styling
- Proper table formatting
- Code blocks
- Multiple colors and fonts
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
import os
from datetime import datetime

def create_professional_pdf():
    """Create a professional PDF from README.md"""
    
    input_file = "README.md"
    output_file = "Assignment_2_Submission.pdf"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return False
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    story = []
    
    # Define custom styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=32
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#555555'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        leading=18
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#2e5c8a'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor('#444444'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=12
    )
    
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_LEFT,
        spaceAfter=6,
        leftIndent=20,
        leading=12
    )
    
    # Read README content
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Add title page
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("Telecom Customer Churn Prediction", title_style))
    story.append(Paragraph("Machine Learning Project", subtitle_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Assignment 2025aa05444", styles['Normal']))
    story.append(Paragraph("BITS Course", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    
    story.append(PageBreak())
    
    # Parse and add content
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Main title
        if line.startswith('# ') and not line.startswith('##'):
            title = line.replace('# ', '')
            story.append(Paragraph(title, heading1_style))
        
        # Heading 2
        elif line.startswith('## '):
            heading = line.replace('## ', '')
            story.append(Paragraph(heading, heading2_style))
        
        # Heading 3
        elif line.startswith('### '):
            heading = line.replace('### ', '')
            story.append(Paragraph(heading, heading3_style))
        
        # Horizontal rule
        elif line == '---':
            story.append(Spacer(1, 0.15*inch))
        
        # Tables
        elif '|' in line:
            table_rows = []
            j = i
            while j < len(lines) and '|' in lines[j]:
                row_text = lines[j].strip()
                if row_text.startswith('|') and row_text.endswith('|'):
                    row = [cell.strip() for cell in row_text.split('|')[1:-1]]
                    # Clean markdown formatting
                    row = [cell.replace('**', '').replace('---', '') for cell in row]
                    if row and any(cell.strip() for cell in row):
                        table_rows.append(row)
                j += 1
            
            if len(table_rows) > 1:
                # Remove separator row if present
                if all('-' in cell or cell == '' for cell in table_rows[1]):
                    table_rows.pop(1)
                
                if len(table_rows) > 1:
                    table = Table(table_rows, colWidths=[1.5*inch]*len(table_rows[0]) if table_rows[0] else None)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
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
            code_text = '\n'.join(code_lines).strip()
            if code_text:
                story.append(Paragraph('<b>Code Example:</b>', styles['Normal']))
                story.append(Paragraph(
                    f'<font face="Courier" size="8" color="#333333">{code_text}</font>',
                    styles['Normal']
                ))
                story.append(Spacer(1, 0.1*inch))
        
        # Bullet points
        elif line.startswith('- '):
            bullet_text = line[2:].strip()
            story.append(Paragraph(f"• {bullet_text}", bullet_style))
        
        # Regular text
        elif line and not line.startswith('#') and not line.startswith('|'):
            story.append(Paragraph(line, normal_style))
        
        else:
            story.append(Spacer(1, 0.05*inch))
        
        i += 1
    
    # Build PDF
    try:
        doc.build(story)
        print(f"✅ Professional PDF created: {output_file}")
        return True
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Professional PDF Generator for Assignment Submission")
    print("=" * 70)
    
    if create_professional_pdf():
        print("\n✅ PDF successfully created!")
        print("   File: Assignment_2_Submission.pdf")
        print("   Ready for submission!")
    else:
        print("\n❌ Failed to create PDF")
    
    print("=" * 70)
