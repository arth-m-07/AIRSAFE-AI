# AIRSAFE-AI  
## Emergency Landing Zone Detection System

---

## Overview

AIRSAFE-AI is a simple AI system that helps determine whether an area is safe for an emergency aircraft landing.

You upload an aerial image of emergency landing surface.

The system:

- Analyzes the terrain using a trained AI model  
- Identifies sand and obstacles  
- Calculates how much safe surface is available  
- Displays a safety score  
- Classifies the area as SAFE or HIGH RISK  

This project is a proof-of-concept prototype for demonstration purposes.

---

## What You Need Before Starting

Make sure you have:

- Python 3.10 or higher installed  
- pip (Python package manager)  
- Internet connection (to install required packages)  
- A web browser (Chrome recommended)  

---

## Project Structure

Your project folder should look like this:

```
AIRSAFE-AI/
│
├── backend/
│   ├── backend.py
│   ├── best_model.pth
│   ├── requirements.txt
│
├── frontend/
│   ├── segmentation.html
│
├── README.md
```

Important:

- The file `best_model.pth` must be inside the `backend` folder.
- Do not rename folders.

---

#  How to Run the Project (Step-by-Step)

Follow these steps carefully.

---

## Step 1: Open Terminal

Navigate to the main project folder:

```bash
cd AIRSAFE-AI
```

---

## Step 2: Start the Backend (AI Model)

Go into the backend folder:

```bash
cd backend
```

Install required libraries:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn backend:app --reload
```

You should see:

```
Application startup complete.
```

The backend is now running at:

```
http://localhost:8000
```

Leave this terminal open.

---

## Step 3: Start the Frontend

Open a new terminal window.

Go to the frontend folder:

```bash
cd AIRSAFE-AI/frontend
```

Start a simple server:

```bash
python -m http.server 5600
```

Open your browser and go to:

```
http://localhost:5600/segmentation.html
```

---

# How to Test the System

1. Upload the aerial image.
2. Click “Run Terrain Analysis”.
3. The system will:
   - Show the segmented image
   - Display a safety score
   - Show whether the area is SAFE or HIGH RISK

---

# How to Reproduce Final Results

To reproduce the final results:

1. Run backend and frontend as described above.
2. Use the aerial images similar to training conditions.
3. Upload them through the interface.
4. Observe the safety score and classification.

Results will vary depending on how much sand is present in the image.

---

# How the Safety Score is Calculated

The AI model identifies different terrain types.

For this prototype:

- Sand and ocean are considered safe.
- Rocks and vegetation are considered unsafe.

The system calculates:

```
Safety Score = Percentage of safe sand area in the image
```

If safety score is greater than 40%:

SAFE LANDING ZONE DETECTED  

If safety score is 40% or less:

HIGH RISK – NO SAFE ZONE  

---

# Expected Output

After running analysis, you will see:

- The original uploaded image
- The segmented terrain overlay
- A safety score (percentage)
- A SAFE or HIGH RISK decision

Example:
<img width="1470" height="956" alt="Screenshot 2026-02-18 at 5 22 41 PM" src="https://github.com/user-attachments/assets/e082fdb5-58a2-4972-a617-b009d78047eb" />

```
Status: HIGH RISK – NO SAFE ZONE
Safety Score: 12.11%

```

---

# Troubleshooting

If nothing loads:

- Make sure backend is running on port 8000.
- Make sure frontend is running on port 5600.
- Make sure `best_model.pth` is inside the backend folder.

If you see connection errors:

- Stop both servers.
- Restart backend first.
- Then restart frontend.

---

# Dependencies Used

This project uses:

- FastAPI (backend server)
- PyTorch (AI model)
- OpenCV (image processing)
- NumPy
- Segmentation Models PyTorch
- HTML / CSS / JavaScript
- Three.js (frontend visuals)

All required packages are listed in:

```
backend/requirements.txt
```

---

# Disclaimer

This system is a research prototype and is not certified for real-world aviation use.  
It is intended for educational and demonstration purposes only.

---


