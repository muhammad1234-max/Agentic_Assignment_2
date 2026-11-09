# Direct Google Colab Execution Guide (Zero Google Drive Upload)
## AI4015 — Agentic AI | Assignment Beta — Question 2
**Student ID:** 23K-0068 | **FAST-NUCES Karachi**

---

### You do NOT need to upload any folder to Google Drive!
The master notebook `Assignment_Beta_Question_no_2_Student_ID.ipynb` has been built to be **100% self-contained**. You only need to upload this single file directly from your computer into Google Colab.

---

### Step 1: Open Google Colab and Upload the Notebook
1. Open your web browser and navigate to: **[https://colab.research.google.com](https://colab.research.google.com)**
2. In the popup window that appears, click the **Upload** tab.
   *(Or click **File** → **Upload notebook** from the top menu).*
3. Click **Browse** and select the file from your computer:
   `C:\Users\hp z book\Desktop\q2 experiment\Assignment_Beta_Question_2\Assignment_Beta_Question_no_2_Student_ID.ipynb`
4. The notebook will open immediately in Google Colab.

---

### Step 2: Select the Free GPU Runtime
1. In the top menu of Google Colab, click **Runtime** → **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (free tier).
3. Click **Save**.
4. In the top-right corner, click **Connect**. Once connected, you will see a green checkmark showing "RAM / Disk".

---

### Step 3: Run All Experiments with One Click
1. Click **Runtime** → **Run all** (or press `Ctrl + F9`).
2. **That's it!** The notebook will automatically:
   - Create all project directories inside Colab's fast virtual machine (`/content/`).
   - Install all required libraries (`transformers`, `peft`, `bitsandbytes`, `python-docx`, etc.).
   - Download/create the exact 2,000-example e-commerce dataset (1,800 train / 200 test, seed 42).
   - Evaluate the **Baseline** (zero-shot base model on 200 test examples).
   - Train and evaluate **LoRA** ($r=8$).
   - Train and evaluate **QLoRA** (4-bit NF4, double quantization, paged AdamW).
   - Train and evaluate **DoRA** (weight magnitude/direction decomposition).
   - Train and evaluate **AdaLoRA** (dynamic SVD rank allocator).
   - Output the **Five-Way Comparison Summary Table**.
   - Perform the **Section 3 Rank Sweep** ($r = 4, 8, 16, 32$) on the winning method.
   - Plot high-resolution publication figures (Tool Accuracy vs Rank, Full Match vs Rank).
   - Perform **Failure Analysis** extracting 5 real failure cases from the test set.
   - Generate the formal **4-page Word Report** (`Question_2_Report.docx`).
   - Bundle all outputs into `Assignment_Beta_Question_2_Submission.zip` and **automatically trigger a download straight to your computer's Downloads folder!**

---

### How Checkpointing & Progress Tracking Work
- **Live Progress Bars:** Hugging Face `Trainer` prints loss and step progress bars every 10 steps.
- **Evaluation Counter:** Prints live test progress:
  `Processed 50/200 | Valid JSON: 49 | Tool Acc: 47 | Full Match: 45`
- **Elapsed Time & VRAM:** Precise peak GPU memory (GB) and training duration (minutes) are recorded and displayed after each stage.
- **Resuming:** If your connection blinks or tab closes, each stage checks for a `.done` marker file. Re-running will skip completed stages and resume immediately.
