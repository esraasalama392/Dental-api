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

MODEL_PATH = "Dental_Final_Model.h5"

CLASS_NAMES = [
    "CANCER | سرطان الفم",
    "Calculus | جير الأسنان",
    "Caries | تسوس",
    "Gingivitis | التهاب اللثة",
    "Healthy | أسنان سليمة",
    "Hypodontia | نقص الأسنان",
    "Mouth Ulcer | قرحة الفم",
    "Tooth Discoloration | تغير لون الأسنان"
]

treatment_database = {
    'CANCER': {
        'Treatment': 'يوصى بضرورة مراجعة طبيب الأسنان أو أخصائي أمراض الفم فوراً.',
        'Tips': 'التشخيص المبكر هو أفضل خطوة لضمان الخطة العلاجية المناسبة.'
    },
    'Calculus': {
        'Treatment': 'إجراء تنظيف عميق للأسنان وإزالة الجير.',
        'Tips': 'يُنصح بتنظيف الأسنان مرتين يومياً واستخدام الخيط الطبي.'
    },
    'Caries': {
        'Treatment': 'حشو السن المتضرر، أو علاج العصب إذا كان التسوس عميقاً.',
        'Tips': 'التقليل من السكريات والمشروبات الغازية.'
    },
    'Gingivitis': {
        'Treatment': 'تنظيف مهني للأسنان، مع استخدام غسول فم مضاد للبكتيريا.',
        'Tips': 'الاهتمام بنظافة اللثة واستخدام فرشاة ناعمة.'
    },
    'Healthy': {
        'Treatment': 'لا يوجد علاج، حالة الفم سليمة.',
        'Tips': 'استمر على روتين العناية بالأسنان المتميز.'
    },
    'Hypodontia': {
        'Treatment': 'استشارة أخصائي تقويم الأسنان.',
        'Tips': 'المتابعة الدورية مع الطبيب.'
    },
    'Mouth Ulcer': {
        'Treatment': 'استخدام جل مطهر أو مسكن موضعي للقرحة.',
        'Tips': 'تجنب الأطعمة الحارة والحامضة.'
    },
    'Tooth Discoloration': {
        'Treatment': 'تبييض الأسنان أو استخدام القشور التجميلية.',
        'Tips': 'التقليل من القهوة والشاي والتدخين.'
    },
    'Unclear': {
        'Treatment': 'الصورة مش واضحة كفاية، حاول تصور من قريب أكتر.',
        'Tips': 'تأكد من الإضاءة وقرب الكاميرا من السن.'
    }
}

print("جاري تحميل الموديل...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ تم تحميل DenseNet121 بنجاح!")

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

        # ✅ لو الـ confidence أقل من 60% مش متأكد
        if confidence < 60:
            return {
                "status": "success",
                "message": "تم التشخيص بنجاح",
                "data": {
                    "disease": "Unclear | صورة غير واضحة",
                    "confidence": round(confidence, 2),
                    "treatment": treatment_database['Unclear']['Treatment'],
                    "tips": treatment_database['Unclear']['Tips'],
                    "blur_value": round(blur_value, 2),
                    "blur_warning": "يرجى التقاط صورة أوضح وأقرب للسن"
                }
            }

        disease = CLASS_NAMES[class_idx]
        disease_key = disease.split(' | ')[0]

        treatment_info = treatment_database.get(disease_key, {
            'Treatment': 'يرجى استشارة طبيب الأسنان',
            'Tips': ''
        })

        blur_warning = "تحذير: الصورة مهزوزة، يرجى التقاط صورة أوضح." if blur_value > 150 else ""

        return {
            "status": "success",
            "message": "تم التشخيص بنجاح",
            "data": {
                "disease": disease,
                "confidence": round(confidence, 2),
                "treatment": treatment_info['Treatment'],
                "tips": treatment_info['Tips'],
                "blur_value": round(blur_value, 2),
                "blur_warning": blur_warning
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة الصورة: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)