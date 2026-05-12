from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO
import cv2

app = FastAPI(title="Dental AI - DenseNet121 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "Dental_Final_Model_V2.h5"

CLASS_NAMES = [
    "CANCER",
    "Calculus",
    "Caries",
    "Gingivitis",
    "Healthy",
    "Hypodontia",
    "Mouth Ulcer",
    "Tooth Discoloration"
]

treatment_database = {
    'CANCER': {
        'Treatment': 'Immediate consultation with a dentist or oral disease specialist is strongly recommended.',
        'Tips': 'Early diagnosis is the best step to ensure the appropriate treatment plan.'
    },
    'Calculus': {
        'Treatment': 'Professional deep dental cleaning and tartar removal is required.',
        'Tips': 'Brush your teeth twice daily and use dental floss regularly.'
    },
    'Caries': {
        'Treatment': 'Filling the affected tooth, or root canal treatment if the decay is deep.',
        'Tips': 'Reduce sugar intake and carbonated drinks.'
    },
    'Gingivitis': {
        'Treatment': 'Professional dental cleaning with antibacterial mouthwash.',
        'Tips': 'Pay attention to gum hygiene and use a soft-bristle toothbrush.'
    },
    'Healthy': {
        'Treatment': 'No treatment needed. Your mouth is in great condition!',
        'Tips': 'Keep up your excellent dental care routine.'
    },
    'Hypodontia': {
        'Treatment': 'Consult an orthodontic specialist for evaluation.',
        'Tips': 'Regular follow-up visits with your dentist are recommended.'
    },
    'Mouth Ulcer': {
        'Treatment': 'Apply antiseptic gel or topical pain relief to the ulcer.',
        'Tips': 'Avoid spicy and acidic foods until healed.'
    },
    'Tooth Discoloration': {
        'Treatment': 'Teeth whitening or cosmetic veneers are recommended.',
        'Tips': 'Reduce coffee, tea, and smoking to prevent further discoloration.'
    },
    'Unclear': {
        'Treatment': 'Image is not clear enough. Please retake a closer photo.',
        'Tips': 'Make sure the lighting is good and the camera is close to the tooth.'
    }
}

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ DenseNet121 loaded successfully!")

def get_blur_value(image_bytes: bytes) -> float:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, PNG files are allowed")

    contents = await file.read()

    try:
        blur_value = get_blur_value(contents)

        img = Image.open(BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array, verbose=0)[0]

        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx] * 100)

        blur_warning = "Warning: Image is blurry, please take a clearer photo." if blur_value > 150 else ""

        # ✅ confidence >= 60% → single result
        if confidence >= 60:
            disease = CLASS_NAMES[class_idx]
            treatment_info = treatment_database.get(disease, {
                'Treatment': 'Please consult a dentist.',
                'Tips': ''
            })
            return {
                "status": "success",
                "message": "Diagnosis completed successfully",
                "data": {
                    "disease": disease,
                    "confidence": round(confidence, 2),
                    "treatment": treatment_info['Treatment'],
                    "tips": treatment_info['Tips'],
                    "blur_value": round(blur_value, 2),
                    "blur_warning": blur_warning,
                    "second_disease": None
                }
            }

        # ✅ confidence < 60% → top 2 results
        else:
            top2_indices = np.argsort(predictions)[::-1][:2]

            disease1 = CLASS_NAMES[top2_indices[0]]
            disease2 = CLASS_NAMES[top2_indices[1]]
            conf1 = round(float(predictions[top2_indices[0]] * 100), 2)
            conf2 = round(float(predictions[top2_indices[1]] * 100), 2)

            treatment1 = treatment_database.get(disease1, {'Treatment': 'Please consult a dentist.', 'Tips': ''})
            treatment2 = treatment_database.get(disease2, {'Treatment': 'Please consult a dentist.', 'Tips': ''})

            return {
                "status": "success",
                "message": "Diagnosis completed successfully",
                "data": {
                    "disease": disease1,
                    "confidence": conf1,
                    "treatment": treatment1['Treatment'],
                    "tips": treatment1['Tips'],
                    "blur_value": round(blur_value, 2),
                    "blur_warning": blur_warning,
                    "second_disease": {
                        "disease": disease2,
                        "confidence": conf2,
                        "treatment": treatment2['Treatment'],
                        "tips": treatment2['Tips']
                    }
                }
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)