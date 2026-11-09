# Hugging Face Spaces & Free GPU Guide (Works 100% in Pakistan)
## AI4015 — Agentic AI | Assignment Beta — Question 2
**Student ID:** 23K-0068 | **FAST-NUCES Karachi**

---

### Why Colab & Kaggle Said "Unavailable"

1. **Why Kaggle didn't give you a GPU:**
   On Kaggle, **GPU and Internet access are locked by default** for new accounts until you complete a one-time SMS verification.
   - **The Fix:** Go to **[kaggle.com/settings](https://www.kaggle.com/settings)** → Scroll down to **Phone verification** → Enter your phone number (+92 3xx...) → Enter the SMS code.
   - Once verified, **dual NVIDIA T4 GPUs (30 hours/week)** and the **Internet toggle** unlock immediately!

2. **Why Google Colab gave an error:**
   Colab's "Cannot connect to GPU backend" error is an **account-level quota block** (your current Google account was temporarily flagged or hit its 12-hour limit).
   - **The Fix:** Open an **Incognito / Private browser window**, log into Colab using a **different Google account** (e.g. personal Gmail instead of university email, or vice versa), and it will assign a free T4 GPU immediately.

---

### How to Run on Hugging Face Spaces (No Phone Verification Needed)

Hugging Face Spaces is accessible in Pakistan without restrictions. You can create a free **JupyterLab Space** on Hugging Face in 2 minutes:

#### Step 1: Create a New Space
1. Log in to **[huggingface.co](https://huggingface.co)**.
2. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**.
3. **Space name:** `peft-experiment`
4. **License:** `apache-2.0`
5. **Select the Space SDK:** Choose **Docker** → Click **Blank**.
6. **Space Hardware:** 
   - By default, it gives you a **Free CPU (2 vCPU, 16 GB RAM)**.
   - *(Optional: If you want blazing-fast GPU speed, you can select the T4 Small GPU for ~$0.60/hr).*
7. Click **Create Space**.

#### Step 2: Add the JupyterLab Dockerfile
1. In your newly created Space, click the **Files** tab.
2. Click **+ Add file** → **Create a new file**.
3. Name the file: `Dockerfile`
4. Paste the following configuration into the file:
   ```dockerfile
   FROM jupyter/scipy-notebook:latest
   USER root
   RUN pip install --no-cache-dir torch transformers peft trl bitsandbytes accelerate datasets python-docx
   USER jovyan
   EXPOSE 7860
   CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=7860", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
   ```
5. Click **Commit new file to main**.

#### Step 3: Launch JupyterLab & Run the Notebook
1. Hugging Face will build the container (takes about 2 minutes). When it finishes, your Space will say **Running**.
2. Click the **App** tab. A full **JupyterLab** interface will appear directly in your browser!
3. Drag and drop `Assignment_Beta_Question_no_2_Student_ID.ipynb` into the left file manager.
4. Double-click the notebook to open it, and click **Run** → **Run All Cells**.
5. When complete, right-click `Assignment_Beta_Question_2_Submission.zip` in the left panel and click **Download**!

