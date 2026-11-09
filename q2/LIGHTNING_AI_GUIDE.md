# Complete Lightning AI Execution Guide (Recommended: Free Instant GPU)
## AI4015 — Agentic AI | Assignment Beta — Question 2
**Student ID:** 23K-0068 | **FAST-NUCES Karachi**

---

### Why Lightning AI?
- **Zero Provisioning Queues:** Launches immediately without the runtime shortages currently affecting Colab and Kaggle.
- **Free Monthly GPU Hours:** 22 free GPU credits every month.
- **NVIDIA T4 or A10G GPUs:** Fast, dedicated modern hardware (16 GB - 24 GB VRAM).
- **Direct Drag-and-Drop:** Native JupyterLab file tree.
- **Persistence:** Studios remain saved even after you pause them.

---

### Step 1: Sign in to Lightning AI
1. Go to **[https://lightning.ai](https://lightning.ai)** in your web browser.
2. Click **Sign In** and choose your Google or GitHub account (instant signup).

---

### Step 2: Create a New Studio
1. From the dashboard, click **+ New Studio** (top right).
2. Give it a name (e.g. `peft-experiment`) or keep the default.
3. The studio interface (JupyterLab + terminal) will load in about 10 seconds.

---

### Step 3: Switch Hardware to GPU
1. In the top-right corner of the Studio interface, look for the hardware badge (it defaults to **CPU**).
2. Click on it and select **T4 GPU** (or **A10G**).
3. The instance will switch to a dedicated NVIDIA GPU within ~15 seconds.

---

### Step 4: Upload Your Notebook
1. In the left-hand sidebar (file manager), drag and drop:
   ```text
   C:\Users\hp z book\Desktop\q2 experiment\Assignment_Beta_Question_2\Assignment_Beta_Question_no_2_Student_ID.ipynb
   ```
2. Double-click the uploaded notebook to open it in the editor.

---

### Step 5: Run the Notebook (One Click)
1. In the notebook toolbar, click the **Run All** button (`>>` icon), or click **Run** → **Run All Cells**.
2. The notebook will execute autonomously:
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
   - Packages all deliverables into `Assignment_Beta_Question_2_Submission.zip`.

---

### Step 6: Download Your Results
When Cell 17 finishes:
1. It will render a clickable blue link in the notebook output:
   `Assignment_Beta_Question_2_Submission.zip`
2. Click that link, or right-click `Assignment_Beta_Question_2_Submission.zip` in the left file tree and click **Download**.
3. All required files (executed notebook, CSVs, high-res plots, and the Word report) are inside the zip!

