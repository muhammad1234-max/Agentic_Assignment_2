# Complete Kaggle Execution Guide (Instant GPU & Zero Setup)
## AI4015 — Agentic AI | Assignment Beta — Question 2
**Student ID:** 23K-0068 | **FAST-NUCES Karachi**

---

### Why Kaggle is Better than Colab:
- **No provisioning delays or runtime shortages** (Colab often exhausts free GPU quotas).
- **Free NVIDIA Tesla T4 x2 (15 GB VRAM each)** or **P100 (16 GB VRAM)**.
- **30 hours per week** of free GPU compute.
- **Direct browser file upload** — no Google Drive required.
- **Clickable download link** for the final submission zip package.

---

### Step 1: Open Kaggle and Create a New Notebook
1. Go to **[https://www.kaggle.com/code](https://www.kaggle.com/code)** (sign in if needed).
2. Click the black **New Notebook** button in the top right.

---

### Step 2: Upload Your Notebook
1. In the Kaggle notebook editor, click **File** (in the top menu bar) → **Import Notebook**.
2. Click the **Upload** tab.
3. Select this file from your computer:
   ```text
   C:\Users\hp z book\Desktop\q2 experiment\Assignment_Beta_Question_2\Assignment_Beta_Question_no_2_Student_ID.ipynb
   ```
4. Click **Import**. The notebook will load immediately.

---

### Step 3: Turn on GPU and Internet (CRITICAL)
In the right sidebar under **Notebook options**:
1. **Accelerator:** Click on the dropdown and select **GPU T4 x2** (or **GPU P100**).
2. **Internet:** Toggle the switch to **Internet on** (This is required so the notebook can download HuggingFace models and packages).
   *(If prompted for phone verification to enable internet, complete the 30-second SMS verification once).*

---

### Step 4: Run the Notebook (One Click)
1. Click **Run All** (the double play button `>>` at the top, or press `Shift + Enter` through cells).
2. **That's it!** The notebook will execute autonomously:
   - Evaluates the **Baseline** (zero-shot base model on 200 test examples).
   - Trains and evaluates **LoRA** ($r=8$).
   - Trains and evaluates **QLoRA** (4-bit NF4, double quantization, paged AdamW).
   - Trains and evaluates **DoRA** (weight magnitude/direction decomposition).
   - Trains and evaluates **AdaLoRA** (dynamic SVD rank allocator).
   - Outputs the **Five-Way Comparison Summary Table**.
   - Performs the **Section 3 Rank Sweep** ($r = 4, 8, 16, 32$) on the winning method.
   - Plots high-resolution publication figures (Tool Accuracy vs Rank, Full Match vs Rank).
   - Extracts and diagnoses **5 real failure cases** from the test set.
   - Generates the formal **4-page Word Report** (`Question_2_Report.docx`).
   - Packages everything into `Assignment_Beta_Question_2_Submission.zip`.

---

### Step 5: Download Your Results
When Cell 17 finishes:
- It will render a clickable blue hyperlink directly in the notebook output:
  `Assignment_Beta_Question_2_Submission.zip`
- Click that link to download the entire zip package directly to your computer.
- Alternatively, on the right sidebar under the **Output** tab, you can click the download button next to `/kaggle/working/Assignment_Beta_Question_2_Submission.zip`.

