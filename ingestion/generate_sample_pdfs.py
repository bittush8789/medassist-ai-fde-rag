import os
import fitz  # PyMuPDF
from pathlib import Path

def create_medical_pdf(file_path: str, title: str, pages_data: list):
    """
    Creates a styled multi-page PDF document using PyMuPDF.
    Each page contains a header, content sections, and a footer with page numbering.
    """
    doc = fitz.open()
    
    for page_idx, page_content in enumerate(pages_data):
        page_num = page_idx + 1
        page = doc.new_page(width=595, height=842)  # A4 size in points
        
        # Header banner
        page.draw_rect(fitz.Rect(40, 30, 555, 65), color=(0.12, 0.45, 0.65), fill=(0.93, 0.96, 0.98))
        page.insert_text(fitz.Point(50, 52), title.upper(), fontsize=11, fontname="helv", color=(0.12, 0.45, 0.65))
        
        y_cursor = 85
        
        for section_title, paragraphs in page_content.items():
            # Section Title
            page.insert_text(fitz.Point(50, y_cursor), section_title, fontsize=12, fontname="helv", color=(0.1, 0.2, 0.4))
            y_cursor += 18
            
            for para in paragraphs:
                rect = fitz.Rect(50, y_cursor, 545, y_cursor + 120)
                rc = page.insert_textbox(rect, para, fontsize=9.5, fontname="helv", color=(0.15, 0.15, 0.15))
                y_cursor += (abs(rc) if rc < 0 else 60) + 12
                
        # Footer
        page.draw_line(fitz.Point(40, 800), fitz.Point(555, 800), color=(0.8, 0.8, 0.8), width=0.5)
        footer_text = f"Document: {Path(file_path).name} | Page {page_num} of {len(pages_data)} | Clinical Knowledge Base"
        page.insert_text(fitz.Point(50, 815), footer_text, fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))
        
    doc.save(file_path)
    doc.close()
    print(f"Generated PDF: {file_path} ({len(pages_data)} pages)")


def generate_all_sample_pdfs(output_dir: str = "medical_documents"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. diabetes_guidelines.pdf
    diabetes_pages = [
        {
            "Section 1: Diagnostic Criteria and Classification": [
                "Type 2 Diabetes Mellitus (T2D) is characterized by progressive loss of adequate beta-cell insulin secretion frequently on the background of insulin resistance.",
                "Diagnostic criteria include: Fasting Plasma Glucose (FPG) >= 126 mg/dL (7.0 mmol/L) after an 8-hour fast; or 2-hour Post-Prandial Glucose >= 200 mg/dL (11.1 mmol/L) during an OGTT (75g); or Glycated Hemoglobin (HbA1c) >= 6.5% (48 mmol/mol); or random plasma glucose >= 200 mg/dL with classic hyperglycemic symptoms (polyuria, polydipsia, unexplained weight loss)."
            ],
            "Section 2: Initial Assessment and Lifestyle Intervention": [
                "All patients diagnosed with T2D must receive Medical Nutrition Therapy (MNT) and diabetes self-management education. Recommend at least 150 minutes of moderate-intensity aerobic physical activity per week, distributed over at least 3 days with no more than 2 consecutive days without exercise."
            ]
        },
        {
            "Section 3: Pharmacological Management": [
                "First-Line Pharmacotherapy: Metformin remains the preferred initial pharmacological agent for the treatment of Type 2 diabetes unless contraindicated. Initial titration starts at 500 mg once or twice daily with meals to reduce gastrointestinal side effects, titrating to a target therapeutic dose of 2000 mg daily.",
                "Cardiorenal Risk Stratification: In patients with established Atherosclerotic Cardiovascular Disease (ASCVD), high risk indicators, Heart Failure (HF), or Chronic Kidney Disease (CKD), SGLT2 inhibitors (e.g., Empagliflozin, Dapagliflozin) or GLP-1 receptor agonists (e.g., Semaglutide, Dulaglutide) with proven CVD/renal benefit are recommended independent of baseline HbA1c.",
                "Dual and Triple Therapy: If glycemic targets are not achieved within 3 months of monotherapy, intensify treatment by adding an agent from a complementary class: SGLT2i, GLP-1 RA, DPP-4 inhibitors, or Pioglitazone."
            ]
        },
        {
            "Section 4: Glycemic Targets and Hypoglycemia Protocols": [
                "Glycemic Goals: An HbA1c goal of < 7.0% (53 mmol/mol) is recommended for most non-pregnant adult patients. More stringent targets (< 6.5%) may be considered for younger patients with short disease duration and low hypoglycemia risk. Less stringent targets (< 7.5% - 8.0%) are appropriate for patients with limited life expectancy, extensive comorbidities, or severe hypoglycemia history.",
                "Hypoglycemia Management ('Rule of 15'): When blood glucose drops below 70 mg/dL (3.9 mmol/L), administer 15-20 grams of fast-acting oral glucose (e.g., 4 glucose tablets, 1/2 cup fruit juice). Recheck blood glucose after 15 minutes. If still < 70 mg/dL, repeat treatment. Once normalized, provide a meal or complex carbohydrate snack."
            ]
        },
        {
            "Section 5: Renal Considerations and Safety Thresholds": [
                "Renal Function Monitoring: Calculate eGFR at least annually in all diabetic patients. Metformin is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min/1.73m2). In patients with eGFR between 30 and 44 mL/min/1.73m2, the maximum recommended Metformin dose is 1000 mg daily.",
                "SGLT2 Inhibitor Renal Limits: Empagliflozin and Dapagliflozin may be initiated down to an eGFR of 20-25 mL/min/1.73m2 for cardiorenal protection, though their glycemic efficacy decreases as eGFR declines."
            ]
        }
    ]
    create_medical_pdf(str(out_path / "diabetes_guidelines.pdf"), "Clinical Practice Guidelines: Type 2 Diabetes", diabetes_pages)

    # 2. clinical_guidelines.pdf
    clinical_pages = [
        {
            "Section 1: Hypertension Classification and Screening": [
                "Hypertension is a major preventable risk factor for cardiovascular disease, stroke, heart failure, and chronic kidney disease.",
                "Blood pressure categories in adults: Normal: Systolic < 120 mmHg and Diastolic < 80 mmHg. Elevated BP: Systolic 120-129 mmHg and Diastolic < 80 mmHg. Stage 1 Hypertension: Systolic 130-139 mmHg or Diastolic 80-89 mmHg. Stage 2 Hypertension: Systolic >= 140 mmHg or Diastolic >= 90 mmHg. Diagnosis requires at least two separate seated office measurements on two separate occasions."
            ]
        },
        {
            "Section 2: Pharmacological Treatment of Hypertension": [
                "First-line antihypertensive drug classes include: Angiotensin-Converting Enzyme (ACE) Inhibitors (e.g., Lisinopril, Enalapril), Angiotensin II Receptor Blockers (ARBs, e.g., Losartan, Valsartan), Dihydropyridine Calcium Channel Blockers (CCBs, e.g., Amlodipine), and Thiazide/Thiazide-like diuretics (e.g., Chlorthalidone, Hydrochlorothiazide).",
                "Monotherapy vs Combination: For Stage 2 Hypertension with BP >= 20/10 mmHg over goal, initiation with two first-line agents of different classes is recommended.",
                "Target Blood Pressure: The primary therapeutic target for adults with confirmed hypertension and known CVD or 10-year ASCVD risk >= 10% is < 130/80 mmHg."
            ]
        },
        {
            "Section 3: Guideline Variations and Target Ambiguities": [
                "Contrasting Guidelines and Target Blood Pressure Thresholds: While recent intensive treatment trials endorse a strict target of < 130/80 mmHg for high-risk cohorts, certain traditional guidelines (such as JNC-8 legacy protocols) recommend a blood pressure threshold of < 140/90 mmHg for general adult populations under age 60, and < 150/90 mmHg for individuals aged 60 and older without diabetes or CKD.",
                "Clinical Resolution of Conflicts: Practitioners must evaluate individualized cardiovascular risk, orthostatic risk, and renal tolerance when setting target thresholds between 130/80 mmHg and 140/90 mmHg."
            ]
        }
    ]
    create_medical_pdf(str(out_path / "clinical_guidelines.pdf"), "National Clinical Guidelines: Cardiovascular & Hypertension", clinical_pages)

    # 3. cardiology_guidelines.pdf
    cardiology_pages = [
        {
            "Section 1: Heart Failure Classification and Diagnosis": [
                "Heart failure is categorized by Left Ventricular Ejection Fraction (LVEF): Heart Failure with Reduced Ejection Fraction (HFrEF: LVEF <= 40%), Heart Failure with Mildly Reduced Ejection Fraction (HFmrEF: LVEF 41-49%), and Heart Failure with Preserved Ejection Fraction (HFpEF: LVEF >= 50%).",
                "New York Heart Association (NYHA) Functional Classes: Class I (No limitation of physical activity), Class II (Slight limitation, comfortable at rest), Class III (Marked limitation of physical activity), Class IV (Unable to carry on any physical activity without discomfort; symptoms at rest)."
            ]
        },
        {
            "Section 2: Guideline-Directed Medical Therapy (GDMT) for HFrEF": [
                "Foundational GDMT consists of Quadruple Therapy to reduce morbidity and mortality:",
                "1. Angiotensin Receptor-Neprilysin Inhibitor (ARNI, Sacubitril/Valsartan) preferred over ACE inhibitors or ARBs.",
                "2. Evidence-Based Beta-Blockers: Bisoprolol, Carvedilol, or sustained-release Metoprolol Succinate.",
                "3. Mineralocorticoid Receptor Antagonists (MRA): Spironolactone or Eplerenone (monitor serum potassium and renal function).",
                "4. SGLT2 Inhibitors: Dapagliflozin (10 mg daily) or Empagliflozin (10 mg daily), recommended for all symptomatic HFrEF patients regardless of diabetes status."
            ]
        },
        {
            "Section 3: Acute Coronary Syndrome (ACS) Management": [
                "Immediate medical therapy for Acute Coronary Syndrome includes chewable Aspirin (162-325 mg), sublingual Nitroglycerin for ischemic pain, and high-intensity Statin therapy (Atorvastatin 80 mg or Rosuvastatin 40 mg).",
                "Dual Antiplatelet Therapy (DAPT): Combine Aspirin with a P2Y12 inhibitor (Ticagrelor 90 mg twice daily or Prasugrel 10 mg once daily preferred over Clopidogrel) for at least 12 months following percutaneous coronary intervention (PCI)."
            ]
        }
    ]
    create_medical_pdf(str(out_path / "cardiology_guidelines.pdf"), "Cardiology Society Consensus: Heart Failure and ACS", cardiology_pages)

    # 4. drug_information.pdf
    drug_pages = [
        {
            "Drug Monograph: Metformin Hydrochloride": [
                "Therapeutic Class: Biguanide Antihyperglycemic Agent.",
                "Indications: First-line management of Type 2 Diabetes Mellitus.",
                "Dosing: Initial 500 mg orally once or twice daily with meals. Increase weekly by 500 mg up to maximum 2000-2550 mg/day in divided doses.",
                "Contraindications and Warnings: Severe renal impairment (eGFR < 30 mL/min/1.73m2), acute metabolic acidosis, severe hepatic impairment, acute congestive heart failure. Black Box Warning for Lactic Acidosis."
            ]
        },
        {
            "Drug Monograph: Lisinopril": [
                "Therapeutic Class: Angiotensin-Converting Enzyme (ACE) Inhibitor.",
                "Indications: Hypertension, Heart Failure with reduced ejection fraction, Post-Myocardial Infarction.",
                "Dosing: Hypertension initial 10 mg once daily, titrating to 20-40 mg once daily. Heart failure initial 2.5-5 mg daily.",
                "Adverse Effects & Warnings: Dry persistent cough, hyperkalemia, angioedema, hypotension. Contraindicated in pregnancy (fetal toxicity) and concomitant use with Aliskiren in diabetic patients."
            ]
        },
        {
            "Drug Monograph: Empagliflozin": [
                "Therapeutic Class: Sodium-Glucose Cotransporter 2 (SGLT2) Inhibitor.",
                "Indications: Type 2 Diabetes, Heart Failure (HFrEF and HFpEF), Chronic Kidney Disease.",
                "Dosing: 10 mg orally once daily in the morning with or without food. May increase to 25 mg once daily for glycemic control.",
                "Adverse Effects & Precautions: Mycotic genital infections, urinary tract infections, volume depletion/hypotension, euglycemic ketoacidosis. Discontinue at least 3 days prior to major surgery to avoid DKA."
            ]
        },
        {
            "Drug Monograph: Atorvastatin Calcium": [
                "Therapeutic Class: HMG-CoA Reductase Inhibitor (Statin).",
                "Indications: Hypercholesterolemia, Primary and Secondary Prevention of Cardiovascular Events.",
                "Dosing: Moderate-intensity: 10-20 mg daily. High-intensity: 40-80 mg once daily.",
                "Adverse Reactions: Myopathy, rhabdomyolysis, elevated transaminases, modest increase in HbA1c/blood glucose. Contraindicated in active liver disease and pregnancy."
            ]
        }
    ]
    create_medical_pdf(str(out_path / "drug_information.pdf"), "Compendium of Drug Monographs & Therapeutics", drug_pages)

    # 5. medical_research.pdf
    research_pages = [
        {
            "Section 1: Landmark Trials in SGLT2 Inhibition and Heart Failure": [
                "EMPA-REG OUTCOME and DAPA-HF Clinical Summary: Large randomized double-blind placebo-controlled trials demonstrated that SGLT2 inhibitors provide profound cardiovascular risk reduction.",
                "In the DAPA-HF trial (4,744 patients with HFrEF), Dapagliflozin reduced the primary composite outcome of cardiovascular death or worsening heart failure by 26% (HR 0.74; 95% CI 0.65-0.85; P < 0.001) compared with placebo. Benefits were statistically identical in patients with and without Type 2 diabetes."
            ]
        },
        {
            "Section 2: Comparative Efficacy of Incretin-Based Therapies": [
                "Head-to-head clinical evaluation of GLP-1 Receptor Agonists (Semaglutide, Tirzepatide) versus SGLT2 inhibitors indicates greater weight loss and HbA1c reduction with GLP-1 RAs (mean HbA1c reduction 1.5% to 2.2% vs 0.8% to 1.1% with SGLT2i).",
                "Conversely, SGLT2 inhibitors demonstrate superior rapid reductions in heart failure hospitalizations and acute decongestion benefits in cardiorenal syndromes."
            ]
        }
    ]
    create_medical_pdf(str(out_path / "medical_research.pdf"), "Medical Research & Clinical Trial Evidence Reviews", research_pages)

if __name__ == "__main__":
    generate_all_sample_pdfs()
