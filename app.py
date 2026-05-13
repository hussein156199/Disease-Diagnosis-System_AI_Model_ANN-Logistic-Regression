# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import warnings
import logging
import os
import joblib

# إخفاء التحذيرات
warnings.filterwarnings("ignore")

# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'disease_diagnosis_secret'
CORS(app)

# المتغيرات العامة
lr_model = None
encoder = None
feature_columns = None

# قائمة الأعراض (نفس القائمة الموجودة في الكود الأصلي)
SYMPTOMS_TRANSLATIONS = {
    'anxiety and nervousness': {'ar': 'قلق وعصبية', 'en': 'anxiety and nervousness'},
    'depression': {'ar': 'اكتئاب', 'en': 'depression'},
    'shortness of breath': {'ar': 'ضيق التنفس', 'en': 'shortness of breath'},
    'depressive or psychotic symptoms': {'ar': 'أعراض اكتئابية أو ذهانية', 'en': 'depressive or psychotic symptoms'},
    'sharp chest pain': {'ar': 'ألم حاد في الصدر', 'en': 'sharp chest pain'},
    'dizziness': {'ar': 'دوار', 'en': 'dizziness'},
    'insomnia': {'ar': 'الأرق', 'en': 'insomnia'},
    'abnormal involuntary movements': {'ar': 'حركات لاإرادية غير طبيعية', 'en': 'abnormal involuntary movements'},
    'chest tightness': {'ar': 'ضيق في الصدر', 'en': 'chest tightness'},
    'palpitations': {'ar': 'خفقان القلب', 'en': 'palpitations'},
    'irregular heartbeat': {'ar': 'عدم انتظام ضربات القلب', 'en': 'irregular heartbeat'},
    'breathing fast': {'ar': 'التنفس السريع', 'en': 'breathing fast'},
    'hoarse voice': {'ar': 'بحة في الصوت', 'en': 'hoarse voice'},
    'sore throat': {'ar': 'التهاب الحلق', 'en': 'sore throat'},
    'difficulty speaking': {'ar': 'صعوبة في التحدث', 'en': 'difficulty speaking'},
    'cough': {'ar': 'السعال', 'en': 'cough'},
    'nasal congestion': {'ar': 'احتقان الأنف', 'en': 'nasal congestion'},
    'throat swelling': {'ar': 'تورم الحلق', 'en': 'throat swelling'},
    'diminished hearing': {'ar': 'ضعف السمع', 'en': 'diminished hearing'},
    'lump in throat': {'ar': 'كتلة في الحلق', 'en': 'lump in throat'},
    'throat feels tight': {'ar': 'شعور بضيق في الحلق', 'en': 'throat feels tight'},
    'difficulty in swallowing': {'ar': 'صعوبة البلع', 'en': 'difficulty in swallowing'},
    'skin swelling': {'ar': 'تورم الجلد', 'en': 'skin swelling'},
    'retention of urine': {'ar': 'احتباس البول', 'en': 'retention of urine'},
    'groin mass': {'ar': 'كتلة في الفخذ', 'en': 'groin mass'},
    'leg pain': {'ar': 'ألم الساق', 'en': 'leg pain'},
    'hip pain': {'ar': 'ألم الورك', 'en': 'hip pain'},
    'suprapubic pain': {'ar': 'ألم فوق العانة', 'en': 'suprapubic pain'},
    'blood in stool': {'ar': 'دم في البراز', 'en': 'blood in stool'},
    'lack of growth': {'ar': 'عدم النمو', 'en': 'lack of growth'},
    'emotional symptoms': {'ar': 'أعراض عاطفية', 'en': 'emotional symptoms'},
    'elbow weakness': {'ar': 'ضعف الكوع', 'en': 'elbow weakness'},
    'back weakness': {'ar': 'ضعف الظهر', 'en': 'back weakness'},
    'pus in sputum': {'ar': 'صديد في البلغم', 'en': 'pus in sputum'},
    'symptoms of the scrotum and testes': {'ar': 'أعراض الصفن والخصيتين', 'en': 'symptoms of the scrotum and testes'},
    'swelling of scrotum': {'ar': 'تورم الصفن', 'en': 'swelling of scrotum'},
    'pain in testicles': {'ar': 'ألم في الخصيتين', 'en': 'pain in testicles'},
    'flatulence': {'ar': 'انتفاخ البطن', 'en': 'flatulence'},
    'pus draining from ear': {'ar': 'صديد يتسرب من الأذن', 'en': 'pus draining from ear'},
    'jaundice': {'ar': 'اليرقان', 'en': 'jaundice'},
    'mass in scrotum': {'ar': 'كتلة في الصفن', 'en': 'mass in scrotum'},
    'white discharge from eye': {'ar': 'إفرازات بيضاء من العين', 'en': 'white discharge from eye'},
    'irritable infant': {'ar': 'رضيع غريب الأطوار', 'en': 'irritable infant'},
    'abusing alcohol': {'ar': 'إساءة استخدام الكحول', 'en': 'abusing alcohol'},
    'fainting': {'ar': 'الإغماء', 'en': 'fainting'},
    'hostile behavior': {'ar': 'سلوك عدواني', 'en': 'hostile behavior'},
    'drug abuse': {'ar': 'إساءة استخدام المخدرات', 'en': 'drug abuse'},
    'sharp abdominal pain': {'ar': 'ألم حاد في البطن', 'en': 'sharp abdominal pain'},
    'feeling ill': {'ar': 'الشعور بالمرض', 'en': 'feeling ill'},
    'vomiting': {'ar': 'القيء', 'en': 'vomiting'},
    'headache': {'ar': 'الصداع', 'en': 'headache'},
    'nausea': {'ar': 'الغثيان', 'en': 'nausea'},
    'diarrhea': {'ar': 'الإسهال', 'en': 'diarrhea'},
    'vaginal itching': {'ar': 'حكة مهبلية', 'en': 'vaginal itching'},
    'vaginal dryness': {'ar': 'جفاف مهبلي', 'en': 'vaginal dryness'},
    'painful urination': {'ar': 'تبول مؤلم', 'en': 'painful urination'},
    'involuntary urination': {'ar': 'تبول لاإرادي', 'en': 'involuntary urination'},
    'pain during intercourse': {'ar': 'ألم أثناء الجماع', 'en': 'pain during intercourse'},
    'frequent urination': {'ar': 'كثرة التبول', 'en': 'frequent urination'},
    'lower abdominal pain': {'ar': 'ألم البطن السفلي', 'en': 'lower abdominal pain'},
    'vaginal discharge': {'ar': 'إفرازات مهبلية', 'en': 'vaginal discharge'},
    'blood in urine': {'ar': 'دم في البول', 'en': 'blood in urine'},
    'hot flashes': {'ar': 'هبات ساخنة', 'en': 'hot flashes'},
    'intermenstrual bleeding': {'ar': 'نزيف بين الدورات', 'en': 'intermenstrual bleeding'},
    'hand or finger pain': {'ar': 'ألم اليد أو الأصابع', 'en': 'hand or finger pain'},
    'wrist pain': {'ar': 'ألم الرسغ', 'en': 'wrist pain'},
    'hand or finger swelling': {'ar': 'تورم اليد أو الأصابع', 'en': 'hand or finger swelling'},
    'arm pain': {'ar': 'ألم الذراع', 'en': 'arm pain'},
    'wrist swelling': {'ar': 'تورم الرسغ', 'en': 'wrist swelling'},
    'arm stiffness or tightness': {'ar': 'تيبس أو ضيق الذراع', 'en': 'arm stiffness or tightness'},
    'arm swelling': {'ar': 'تورم الذراع', 'en': 'arm swelling'},
    'hand or finger stiffness or tightness': {'ar': 'تيبس أو ضيق اليد أو الأصابع',
                                              'en': 'hand or finger stiffness or tightness'},
    'wrist stiffness or tightness': {'ar': 'تيبس أو ضيق الرسغ', 'en': 'wrist stiffness or tightness'},
    'lip swelling': {'ar': 'تورم الشفاه', 'en': 'lip swelling'},
    'toothache': {'ar': 'ألم الأسنان', 'en': 'toothache'},
    'abnormal appearing skin': {'ar': 'جلد غير طبيعي المظهر', 'en': 'abnormal appearing skin'},
    'skin lesion': {'ar': 'آفة جلدية', 'en': 'skin lesion'},
    'acne or pimples': {'ar': 'حب الشباب أو البثور', 'en': 'acne or pimples'},
    'dry lips': {'ar': 'جفاف الشفاه', 'en': 'dry lips'},
    'facial pain': {'ar': 'ألم الوجه', 'en': 'facial pain'},
    'mouth ulcer': {'ar': 'قرحة الفم', 'en': 'mouth ulcer'},
    'skin growth': {'ar': 'نمو جلدي', 'en': 'skin growth'},
    'eye deviation': {'ar': 'انحراف العين', 'en': 'eye deviation'},
    'diminished vision': {'ar': 'ضعف الرؤية', 'en': 'diminished vision'},
    'double vision': {'ar': 'رؤية مزدوجة', 'en': 'double vision'},
    'cross-eyed': {'ar': 'العين الحول', 'en': 'cross-eyed'},
    'symptoms of eye': {'ar': 'أعراض العين', 'en': 'symptoms of eye'},
    'pain in eye': {'ar': 'ألم في العين', 'en': 'pain in eye'},
    'eye moves abnormally': {'ar': 'حركات العين غير الطبيعية', 'en': 'eye moves abnormally'},
    'abnormal movement of eyelid': {'ar': 'حركة غير طبيعية للجفن', 'en': 'abnormal movement of eyelid'},
    'foreign body sensation in eye': {'ar': 'الشعور بجسم غريب في العين', 'en': 'foreign body sensation in eye'},
    'irregular appearing scalp': {'ar': 'فروة رأس غير منتظمة المظهر', 'en': 'irregular appearing scalp'},
    'swollen lymph nodes': {'ar': 'تضخم الغدد الليمفاوية', 'en': 'swollen lymph nodes'},
    'back pain': {'ar': 'ألم الظهر', 'en': 'back pain'},
    'neck pain': {'ar': 'ألم الرقبة', 'en': 'neck pain'},
    'low back pain': {'ar': 'ألم أسفل الظهر', 'en': 'low back pain'},
    'pain of the anus': {'ar': 'ألم الشرج', 'en': 'pain of the anus'},
    'pain during pregnancy': {'ar': 'ألم أثناء الحمل', 'en': 'pain during pregnancy'},
    'pelvic pain': {'ar': 'ألم الحوض', 'en': 'pelvic pain'},
    'impotence': {'ar': 'العجز الجنسي', 'en': 'impotence'},
    'infant spitting up': {'ar': 'قيء الرضيع', 'en': 'infant spitting up'},
    'vomiting blood': {'ar': 'قيء الدم', 'en': 'vomiting blood'},
    'regurgitation': {'ar': 'الارتجاع', 'en': 'regurgitation'},
    'burning abdominal pain': {'ar': 'ألم حرقة في البطن', 'en': 'burning abdominal pain'},
    'restlessness': {'ar': 'الأرق والقلق', 'en': 'restlessness'},
    'symptoms of infants': {'ar': 'أعراض الرضع', 'en': 'symptoms of infants'},
    'wheezing': {'ar': 'الأزيز', 'en': 'wheezing'},
    'peripheral edema': {'ar': 'وذمة طرفية', 'en': 'peripheral edema'},
    'neck mass': {'ar': 'كتلة في الرقبة', 'en': 'neck mass'},
    'ear pain': {'ar': 'ألم الأذن', 'en': 'ear pain'},
    'jaw swelling': {'ar': 'تورم الفك', 'en': 'jaw swelling'},
    'mouth dryness': {'ar': 'جفاف الفم', 'en': 'mouth dryness'},
    'neck swelling': {'ar': 'تورم الرقبة', 'en': 'neck swelling'},
    'knee pain': {'ar': 'ألم الركبة', 'en': 'knee pain'},
    'foot or toe pain': {'ar': 'ألم القدم أو الأصابع', 'en': 'foot or toe pain'},
    'bowlegged or knock-kneed': {'ar': 'ساق مقوسة أو متقاطعة', 'en': 'bowlegged or knock-kneed'},
    'ankle pain': {'ar': 'ألم الكاحل', 'en': 'ankle pain'},
    'bones are painful': {'ar': 'ألم العظام', 'en': 'bones are painful'},
    'knee weakness': {'ar': 'ضعف الركبة', 'en': 'knee weakness'},
    'elbow pain': {'ar': 'ألم الكوع', 'en': 'elbow pain'},
    'knee swelling': {'ar': 'تورم الركبة', 'en': 'knee swelling'},
    'skin moles': {'ar': 'شامات جلدية', 'en': 'skin moles'},
    'knee lump or mass': {'ar': 'كتلة أو تورم في الركبة', 'en': 'knee lump or mass'},
    'weight gain': {'ar': 'زيادة الوزن', 'en': 'weight gain'},
    'problems with movement': {'ar': 'مشاكل الحركة', 'en': 'problems with movement'},
    'knee stiffness or tightness': {'ar': 'تيبس أو ضيق الركبة', 'en': 'knee stiffness or tightness'},
    'leg swelling': {'ar': 'تورم الساق', 'en': 'leg swelling'},
    'foot or toe swelling': {'ar': 'تورم القدم أو الأصابع', 'en': 'foot or toe swelling'},
    'heartburn': {'ar': 'حموضة المعدة', 'en': 'heartburn'},
    'smoking problems': {'ar': 'مشاكل التدخين', 'en': 'smoking problems'},
    'muscle pain': {'ar': 'ألم عضلي', 'en': 'muscle pain'},
    'infant feeding problem': {'ar': 'مشكلة تغذية الرضيع', 'en': 'infant feeding problem'},
    'recent weight loss': {'ar': 'فقدان وزن مؤخر', 'en': 'recent weight loss'},
    'problems with shape or size of breast': {'ar': 'مشاكل في شكل أو حجم الثدي',
                                              'en': 'problems with shape or size of breast'},
    'underweight': {'ar': 'نقص الوزن', 'en': 'underweight'},
    'difficulty eating': {'ar': 'صعوبة الأكل', 'en': 'difficulty eating'},
    'scanty menstrual flow': {'ar': 'دورة شهرية خفيفة', 'en': 'scanty menstrual flow'},
    'vaginal pain': {'ar': 'ألم مهبلي', 'en': 'vaginal pain'},
    'vaginal redness': {'ar': 'احمرار مهبلي', 'en': 'vaginal redness'},
    'vulvar irritation': {'ar': 'تهيج الفرج', 'en': 'vulvar irritation'},
    'weakness': {'ar': 'ضعف عام', 'en': 'weakness'},
    'decreased heart rate': {'ar': 'انخفاض ضربات القلب', 'en': 'decreased heart rate'},
    'increased heart rate': {'ar': 'زيادة ضربات القلب', 'en': 'increased heart rate'},
    'bleeding or discharge from nipple': {'ar': 'نزيف أو إفراز من الحلمة', 'en': 'bleeding or discharge from nipple'},
    'ringing in ear': {'ar': 'طنين في الأذن', 'en': 'ringing in ear'},
    'plugged feeling in ear': {'ar': 'الشعور بانسداد الأذن', 'en': 'plugged feeling in ear'},
    'itchy ear(s)': {'ar': 'حكة في الأذن', 'en': 'itchy ear(s)'},
    'frontal headache': {'ar': 'صداع أمامي', 'en': 'frontal headache'},
    'fluid in ear': {'ar': 'سائل في الأذن', 'en': 'fluid in ear'},
    'neck stiffness or tightness': {'ar': 'تيبس أو ضيق الرقبة', 'en': 'neck stiffness or tightness'},
    'spots or clouds in vision': {'ar': 'بقع أو غيوم في الرؤية', 'en': 'spots or clouds in vision'},
    'eye redness': {'ar': 'احمرار العين', 'en': 'eye redness'},
    'lacrimation': {'ar': 'دماع العين', 'en': 'lacrimation'},
    'itchiness of eye': {'ar': 'حكة العين', 'en': 'itchiness of eye'},
    'blindness': {'ar': 'العمى', 'en': 'blindness'},
    'eye burns or stings': {'ar': 'حرق أو وخز العين', 'en': 'eye burns or stings'},
    'itchy eyelid': {'ar': 'حكة الجفن', 'en': 'itchy eyelid'},
    'feeling cold': {'ar': 'الشعور بالبرد', 'en': 'feeling cold'},
    'decreased appetite': {'ar': 'نقص الشهية', 'en': 'decreased appetite'},
    'excessive appetite': {'ar': 'شهية مفرطة', 'en': 'excessive appetite'},
    'excessive anger': {'ar': 'غضب مفرط', 'en': 'excessive anger'},
    'loss of sensation': {'ar': 'فقدان الحس', 'en': 'loss of sensation'},
    'focal weakness': {'ar': 'ضعف موضعي', 'en': 'focal weakness'},
    'slurring words': {'ar': 'تأتأة الكلام', 'en': 'slurring words'},
    'symptoms of the face': {'ar': 'أعراض الوجه', 'en': 'symptoms of the face'},
    'disturbance of memory': {'ar': 'اضطراب الذاكرة', 'en': 'disturbance of memory'},
    'paresthesia': {'ar': 'تنميل وخدر', 'en': 'paresthesia'},
    'side pain': {'ar': 'ألم الجانب', 'en': 'side pain'},
    'fever': {'ar': 'الحمى', 'en': 'fever'},
    'shoulder pain': {'ar': 'ألم الكتف', 'en': 'shoulder pain'},
    'shoulder stiffness or tightness': {'ar': 'تيبس أو ضيق الكتف', 'en': 'shoulder stiffness or tightness'},
    'shoulder weakness': {'ar': 'ضعف الكتف', 'en': 'shoulder weakness'},
    'arm cramps or spasms': {'ar': 'تشنجات الذراع', 'en': 'arm cramps or spasms'},
    'shoulder swelling': {'ar': 'تورم الكتف', 'en': 'shoulder swelling'},
    'tongue lesions': {'ar': 'آفات اللسان', 'en': 'tongue lesions'},
    'leg cramps or spasms': {'ar': 'تشنجات الساق', 'en': 'leg cramps or spasms'},
    'abnormal appearing tongue': {'ar': 'لسان غير طبيعي المظهر', 'en': 'abnormal appearing tongue'},
    'ache all over': {'ar': 'ألم في كل مكان', 'en': 'ache all over'},
    'lower body pain': {'ar': 'ألم الجسم السفلي', 'en': 'lower body pain'},
    'problems during pregnancy': {'ar': 'مشاكل أثناء الحمل', 'en': 'problems during pregnancy'},
    'spotting or bleeding during pregnancy': {'ar': 'نزيف أو تنقيط أثناء الحمل',
                                              'en': 'spotting or bleeding during pregnancy'},
    'cramps and spasms': {'ar': 'تشنجات وتقلصات', 'en': 'cramps and spasms'},
    'upper abdominal pain': {'ar': 'ألم البطن العلوي', 'en': 'upper abdominal pain'},
    'stomach bloating': {'ar': 'انتفاخ المعدة', 'en': 'stomach bloating'},
    'changes in stool appearance': {'ar': 'تغيرات في مظهر البراز', 'en': 'changes in stool appearance'},
    'unusual color or odor to urine': {'ar': 'لون أو رائحة غير عادية للبول', 'en': 'unusual color or odor to urine'},
    'kidney mass': {'ar': 'كتلة في الكلى', 'en': 'kidney mass'},
    'swollen abdomen': {'ar': 'انتفاخ البطن', 'en': 'swollen abdomen'},
    'symptoms of prostate': {'ar': 'أعراض البروستاتا', 'en': 'symptoms of prostate'},
    'leg stiffness or tightness': {'ar': 'تيبس أو ضيق الساق', 'en': 'leg stiffness or tightness'},
    'difficulty breathing': {'ar': 'صعوبة التنفس', 'en': 'difficulty breathing'},
    'rib pain': {'ar': 'ألم الأضلاع', 'en': 'rib pain'},
    'joint pain': {'ar': 'ألم المفاصل', 'en': 'joint pain'},
    'muscle stiffness or tightness': {'ar': 'تيبس أو ضيق العضلات', 'en': 'muscle stiffness or tightness'},
    'pallor': {'ar': 'شحوب', 'en': 'pallor'},
    'hand or finger lump or mass': {'ar': 'كتلة في اليد أو الأصابع', 'en': 'hand or finger lump or mass'},
    'chills': {'ar': 'قشعريرة', 'en': 'chills'},
    'groin pain': {'ar': 'ألم الفخذ', 'en': 'groin pain'},
    'fatigue': {'ar': 'التعب والإرهاق', 'en': 'fatigue'},
    'abdominal distention': {'ar': 'انتفاخ البطن', 'en': 'abdominal distention'},
    'regurgitation.1': {'ar': 'الارتجاع', 'en': 'regurgitation'},
    'symptoms of the kidneys': {'ar': 'أعراض الكلى', 'en': 'symptoms of the kidneys'},
    'melena': {'ar': 'براز أسود (دم محروق)', 'en': 'melena'},
    'flushing': {'ar': 'احمرار الوجه', 'en': 'flushing'},
    'coughing up sputum': {'ar': 'السعال مع البلغم', 'en': 'coughing up sputum'},
    'seizures': {'ar': 'نوبات تشنجية', 'en': 'seizures'},
    'delusions or hallucinations': {'ar': 'أوهام أو هلوسات', 'en': 'delusions or hallucinations'},
    'shoulder cramps or spasms': {'ar': 'تشنجات الكتف', 'en': 'shoulder cramps or spasms'},
    'joint stiffness or tightness': {'ar': 'تيبس أو ضيق المفاصل', 'en': 'joint stiffness or tightness'},
    'pain or soreness of breast': {'ar': 'ألم أو احتقان الثدي', 'en': 'pain or soreness of breast'},
    'excessive urination at night': {'ar': 'كثرة التبول ليلاً', 'en': 'excessive urination at night'},
    'bleeding from eye': {'ar': 'نزيف من العين', 'en': 'bleeding from eye'},
    'rectal bleeding': {'ar': 'نزيف المستقيم', 'en': 'rectal bleeding'},
    'constipation': {'ar': 'الإمساك', 'en': 'constipation'},
    'temper problems': {'ar': 'مشاكل المزاج', 'en': 'temper problems'},
    'coryza': {'ar': 'الزكام', 'en': 'coryza'},
    'wrist weakness': {'ar': 'ضعف الرسغ', 'en': 'wrist weakness'},
    'eye strain': {'ar': 'إجهاد العين', 'en': 'eye strain'},
    'hemoptysis': {'ar': 'نزيف في البلغم', 'en': 'hemoptysis'},
    'lymphedema': {'ar': 'وذمة لمفاوية', 'en': 'lymphedema'},
    'skin on leg or foot looks infected': {'ar': 'جلد الساق أو القدم يبدو مصاباً',
                                           'en': 'skin on leg or foot looks infected'},
    'allergic reaction': {'ar': 'رد فعل تحسسي', 'en': 'allergic reaction'},
    'congestion in chest': {'ar': 'احتقان في الصدر', 'en': 'congestion in chest'},
    'muscle swelling': {'ar': 'تورم العضلات', 'en': 'muscle swelling'},
    'pus in urine': {'ar': 'صديد في البول', 'en': 'pus in urine'},
    'abnormal size or shape of ear': {'ar': 'حجم أو شكل غير طبيعي للأذن', 'en': 'abnormal size or shape of ear'},
    'low back weakness': {'ar': 'ضعف أسفل الظهر', 'en': 'low back weakness'},
    'sleepiness': {'ar': 'النعاس', 'en': 'sleepiness'},
    'apnea': {'ar': 'توقف التنفس أثناء النوم', 'en': 'apnea'},
    'abnormal breathing sounds': {'ar': 'أصوات التنفس غير الطبيعية', 'en': 'abnormal breathing sounds'},
    'excessive growth': {'ar': 'نمو مفرط', 'en': 'excessive growth'},
    'elbow cramps or spasms': {'ar': 'تشنجات الكوع', 'en': 'elbow cramps or spasms'},
    'feeling hot and cold': {'ar': 'الشعور بالحار والبارد', 'en': 'feeling hot and cold'},
    'blood clots during menstrual periods': {'ar': 'جلطات دموية أثناء الدورة الشهرية',
                                             'en': 'blood clots during menstrual periods'},
    'absence of menstruation': {'ar': 'غياب الدورة الشهرية', 'en': 'absence of menstruation'},
    'pulling at ears': {'ar': 'شد الأذن', 'en': 'pulling at ears'},
    'gum pain': {'ar': 'ألم اللثة', 'en': 'gum pain'},
    'redness in ear': {'ar': 'احمرار الأذن', 'en': 'redness in ear'},
    'fluid retention': {'ar': 'احتباس السوائل', 'en': 'fluid retention'},
    'flu-like syndrome': {'ar': 'متلازمة تشبه الإنفلونزا', 'en': 'flu-like syndrome'},
    'sinus congestion': {'ar': 'احتقان الجيوب الأنفية', 'en': 'sinus congestion'},
    'painful sinuses': {'ar': 'جيوب أنفية مؤلمة', 'en': 'painful sinuses'},
    'fears and phobias': {'ar': 'الخوف والرهاب', 'en': 'fears and phobias'},
    'recent pregnancy': {'ar': 'حمل مؤخر', 'en': 'recent pregnancy'},
    'uterine contractions': {'ar': 'تقلصات الرحم', 'en': 'uterine contractions'},
    'burning chest pain': {'ar': 'حرقة في الصدر', 'en': 'burning chest pain'},
    'back cramps or spasms': {'ar': 'تشنجات الظهر', 'en': 'back cramps or spasms'},
    'stiffness all over': {'ar': 'تيبس شامل', 'en': 'stiffness all over'},
    'muscle cramps, contractures, or spasms': {'ar': 'تشنجات أو تقلصات العضلات',
                                               'en': 'muscle cramps, contractures, or spasms'},
    'low back cramps or spasms': {'ar': 'تشنجات أسفل الظهر', 'en': 'low back cramps or spasms'},
    'back mass or lump': {'ar': 'كتلة في الظهر', 'en': 'back mass or lump'},
    'nosebleed': {'ar': 'نزيف الأنف', 'en': 'nosebleed'},
    'long menstrual periods': {'ar': 'دورة شهرية طويلة', 'en': 'long menstrual periods'},
    'heavy menstrual flow': {'ar': 'نزيف حيضي غزير', 'en': 'heavy menstrual flow'},
    'unpredictable menstruation': {'ar': 'دورة شهرية غير منتظمة', 'en': 'unpredictable menstruation'},
    'painful menstruation': {'ar': 'دورة شهرية مؤلمة', 'en': 'painful menstruation'},
    'infertility': {'ar': 'العقم', 'en': 'infertility'},
    'frequent menstruation': {'ar': 'دورة شهرية متكررة', 'en': 'frequent menstruation'},
    'sweating': {'ar': 'التعرق', 'en': 'sweating'},
    'mass on eyelid': {'ar': 'كتلة على الجفن', 'en': 'mass on eyelid'},
    'swollen eye': {'ar': 'تورم العين', 'en': 'swollen eye'},
    'eyelid swelling': {'ar': 'تورم الجفن', 'en': 'eyelid swelling'},
    'eyelid lesion or rash': {'ar': 'آفة أو طفح على الجفن', 'en': 'eyelid lesion or rash'},
    'unwanted hair': {'ar': 'شعر غير مرغوب', 'en': 'unwanted hair'},
    'symptoms of bladder': {'ar': 'أعراض المثانة', 'en': 'symptoms of bladder'},
    'irregular appearing nails': {'ar': 'أظافر غير منتظمة المظهر', 'en': 'irregular appearing nails'},
    'itching of skin': {'ar': 'حكة الجلد', 'en': 'itching of skin'},
    'hurts to breath': {'ar': 'التنفس مؤلم', 'en': 'hurts to breath'},
    'nailbiting': {'ar': 'قضم الأظافر', 'en': 'nailbiting'},
    'skin dryness, peeling, scaliness, or roughness': {'ar': 'جفاف أو تقشر الجلد',
                                                       'en': 'skin dryness, peeling, scaliness, or roughness'},
    'skin on arm or hand looks infected': {'ar': 'جلد الذراع أو اليد يبدو مصاباً',
                                           'en': 'skin on arm or hand looks infected'},
    'skin irritation': {'ar': 'تهيج الجلد', 'en': 'skin irritation'},
    'itchy scalp': {'ar': 'حكة فروة الرأس', 'en': 'itchy scalp'},
    'hip swelling': {'ar': 'تورم الورك', 'en': 'hip swelling'},
    'incontinence of stool': {'ar': 'سلس البراز', 'en': 'incontinence of stool'},
    'foot or toe cramps or spasms': {'ar': 'تشنجات القدم أو الأصابع', 'en': 'foot or toe cramps or spasms'},
    'warts': {'ar': 'الثآليل', 'en': 'warts'},
    'bumps on penis': {'ar': 'نتوءات على العضو الذكري', 'en': 'bumps on penis'},
    'too little hair': {'ar': 'شعر قليل جداً', 'en': 'too little hair'},
    'foot or toe lump or mass': {'ar': 'كتلة في القدم أو الأصابع', 'en': 'foot or toe lump or mass'},
    'skin rash': {'ar': 'طفح جلدي', 'en': 'skin rash'},
    'mass or swelling around the anus': {'ar': 'كتلة أو تورم حول الشرج', 'en': 'mass or swelling around the anus'},
    'low back swelling': {'ar': 'تورم أسفل الظهر', 'en': 'low back swelling'},
    'ankle swelling': {'ar': 'تورم الكاحل', 'en': 'ankle swelling'},
    'hip lump or mass': {'ar': 'كتلة في الورك', 'en': 'hip lump or mass'},
    'drainage in throat': {'ar': 'صرف في الحلق', 'en': 'drainage in throat'},
    'dry or flaky scalp': {'ar': 'فروة رأس جافة أو متقشرة', 'en': 'dry or flaky scalp'},
    'premenstrual tension or irritability': {'ar': 'توتر ما قبل الحيض أو التهيج',
                                             'en': 'premenstrual tension or irritability'},
    'feeling hot': {'ar': 'الشعور بالحرارة', 'en': 'feeling hot'},
    'feet turned in': {'ar': 'قدم متحولة للداخل', 'en': 'feet turned in'},
    'foot or toe stiffness or tightness': {'ar': 'تيبس أو ضيق القدم أو الأصابع',
                                           'en': 'foot or toe stiffness or tightness'},
    'pelvic pressure': {'ar': 'ضغط الحوض', 'en': 'pelvic pressure'},
    'elbow swelling': {'ar': 'تورم الكوع', 'en': 'elbow swelling'},
    'elbow stiffness or tightness': {'ar': 'تيبس أو ضيق الكوع', 'en': 'elbow stiffness or tightness'},
    'early or late onset of menopause': {'ar': 'سن اليأس المبكر أو المتأخر', 'en': 'early or late onset of menopause'},
    'mass on ear': {'ar': 'كتلة على الأذن', 'en': 'mass on ear'},
    'bleeding from ear': {'ar': 'نزيف من الأذن', 'en': 'bleeding from ear'},
    'hand or finger weakness': {'ar': 'ضعف اليد أو الأصابع', 'en': 'hand or finger weakness'},
    'low self-esteem': {'ar': 'تقدير ذات منخفض', 'en': 'low self-esteem'},
    'throat irritation': {'ar': 'تهيج الحلق', 'en': 'throat irritation'},
    'itching of the anus': {'ar': 'حكة الشرج', 'en': 'itching of the anus'},
    'swollen or red tonsils': {'ar': 'لوزات متورمة أو حمراء', 'en': 'swollen or red tonsils'},
    'irregular belly button': {'ar': 'سرة غير منتظمة', 'en': 'irregular belly button'},
    'swollen tongue': {'ar': 'تورم اللسان', 'en': 'swollen tongue'},
    'lip sore': {'ar': 'قرحة على الشفة', 'en': 'lip sore'},
    'vulvar sore': {'ar': 'قرحة على الفرج', 'en': 'vulvar sore'},
    'hip stiffness or tightness': {'ar': 'تيبس أو ضيق الورك', 'en': 'hip stiffness or tightness'},
    'mouth pain': {'ar': 'ألم الفم', 'en': 'mouth pain'},
    'arm weakness': {'ar': 'ضعف الذراع', 'en': 'arm weakness'},
    'leg lump or mass': {'ar': 'كتلة في الساق', 'en': 'leg lump or mass'},
    'disturbance of smell or taste': {'ar': 'اضطراب حاسة الشم أو الذوق', 'en': 'disturbance of smell or taste'},
    'discharge in stools': {'ar': 'إفراز في البراز', 'en': 'discharge in stools'},
    'penis pain': {'ar': 'ألم العضو الذكري', 'en': 'penis pain'},
    'loss of sex drive': {'ar': 'فقدان الرغبة الجنسية', 'en': 'loss of sex drive'},
    'obsessions and compulsions': {'ar': 'وسواس قهري', 'en': 'obsessions and compulsions'},
    'antisocial behavior': {'ar': 'سلوك معادي للمجتمع', 'en': 'antisocial behavior'},
    'neck cramps or spasms': {'ar': 'تشنجات الرقبة', 'en': 'neck cramps or spasms'},
    'pupils unequal': {'ar': 'الحدقات غير متساوية', 'en': 'pupils unequal'},
    'poor circulation': {'ar': 'ضعف الدورة الدموية', 'en': 'poor circulation'},
    'thirst': {'ar': 'العطش', 'en': 'thirst'},
    'sleepwalking': {'ar': 'المشي أثناء النوم', 'en': 'sleepwalking'},
    'skin oiliness': {'ar': 'دهنية الجلد', 'en': 'skin oiliness'},
    'sneezing': {'ar': 'العطس', 'en': 'sneezing'},
    'bladder mass': {'ar': 'كتلة في المثانة', 'en': 'bladder mass'},
    'knee cramps or spasms': {'ar': 'تشنجات الركبة', 'en': 'knee cramps or spasms'},
    'premature ejaculation': {'ar': 'القذف المبكر', 'en': 'premature ejaculation'},
    'leg weakness': {'ar': 'ضعف الساق', 'en': 'leg weakness'},
    'posture problems': {'ar': 'مشاكل الوضعية', 'en': 'posture problems'},
    'bleeding in mouth': {'ar': 'نزيف في الفم', 'en': 'bleeding in mouth'},
    'tongue bleeding': {'ar': 'نزيف اللسان', 'en': 'tongue bleeding'},
    'change in skin mole size or color': {'ar': 'تغير حجم أو لون الشامة', 'en': 'change in skin mole size or color'},
    'penis redness': {'ar': 'احمرار العضو الذكري', 'en': 'penis redness'},
    'penile discharge': {'ar': 'إفراز من العضو الذكري', 'en': 'penile discharge'},
    'shoulder lump or mass': {'ar': 'كتلة في الكتف', 'en': 'shoulder lump or mass'},
    'polyuria': {'ar': 'كثرة التبول', 'en': 'polyuria'},
    'cloudy eye': {'ar': 'عين غائمة', 'en': 'cloudy eye'},
    'hysterical behavior': {'ar': 'سلوك هستيري', 'en': 'hysterical behavior'},
    'arm lump or mass': {'ar': 'كتلة في الذراع', 'en': 'arm lump or mass'},
    'nightmares': {'ar': 'الكوابيس', 'en': 'nightmares'},
    'bleeding gums': {'ar': 'نزيف اللثة', 'en': 'bleeding gums'},
    'pain in gums': {'ar': 'ألم اللثة', 'en': 'pain in gums'},
    'bedwetting': {'ar': 'تبول ليلي لاإرادي', 'en': 'bedwetting'},
    'diaper rash': {'ar': 'طفح حفاضات', 'en': 'diaper rash'},
    'lump or mass of breast': {'ar': 'كتلة في الثدي', 'en': 'lump or mass of breast'},
    'vaginal bleeding after menopause': {'ar': 'نزيف مهبلي بعد انقطاع الطمث', 'en': 'vaginal bleeding after menopause'},
    'infrequent menstruation': {'ar': 'دورة شهرية نادرة', 'en': 'infrequent menstruation'},
    'mass on vulva': {'ar': 'كتلة على الفرج', 'en': 'mass on vulva'},
    'jaw pain': {'ar': 'ألم الفك', 'en': 'jaw pain'},
    'itching of scrotum': {'ar': 'حكة الصفن', 'en': 'itching of scrotum'},
    'postpartum problems of the breast': {'ar': 'مشاكل الثدي بعد الولادة', 'en': 'postpartum problems of the breast'},
    'eyelid retracted': {'ar': 'جفن منسحب', 'en': 'eyelid retracted'},
    'hesitancy': {'ar': 'تردد التبول', 'en': 'hesitancy'},
    'elbow lump or mass': {'ar': 'كتلة في الكوع', 'en': 'elbow lump or mass'},
    'muscle weakness': {'ar': 'ضعف العضلات', 'en': 'muscle weakness'},
    'throat redness': {'ar': 'احمرار الحلق', 'en': 'throat redness'},
    'joint swelling': {'ar': 'تورم المفاصل', 'en': 'joint swelling'},
    'tongue pain': {'ar': 'ألم اللسان', 'en': 'tongue pain'},
    'redness in or around nose': {'ar': 'احمرار الأنف أو حوله', 'en': 'redness in or around nose'},
    'wrinkles on skin': {'ar': 'تجاعيد على الجلد', 'en': 'wrinkles on skin'},
    'foot or toe weakness': {'ar': 'ضعف القدم أو الأصابع', 'en': 'foot or toe weakness'},
    'hand or finger cramps or spasms': {'ar': 'تشنجات اليد أو الأصابع', 'en': 'hand or finger cramps or spasms'},
    'back stiffness or tightness': {'ar': 'تيبس أو ضيق الظهر', 'en': 'back stiffness or tightness'},
    'wrist lump or mass': {'ar': 'كتلة في الرسغ', 'en': 'wrist lump or mass'},
    'skin pain': {'ar': 'ألم الجلد', 'en': 'skin pain'},
    'low back stiffness or tightness': {'ar': 'تيبس أو ضيق أسفل الظهر', 'en': 'low back stiffness or tightness'},
    'low urine output': {'ar': 'انخفاض إنتاج البول', 'en': 'low urine output'},
    'skin on head or neck looks infected': {'ar': 'جلد الرأس أو الرقبة يبدو مصاباً',
                                            'en': 'skin on head or neck looks infected'},
    'stuttering or stammering': {'ar': 'التأتأة', 'en': 'stuttering or stammering'},
    'problems with orgasm': {'ar': 'مشاكل الرعشة', 'en': 'problems with orgasm'},
    'nose deformity': {'ar': 'تشويه الأنف', 'en': 'nose deformity'},
    'lump over jaw': {'ar': 'كتلة فوق الفك', 'en': 'lump over jaw'},
    'sore in nose': {'ar': 'قرحة في الأنف', 'en': 'sore in nose'},
    'hip weakness': {'ar': 'ضعف الورك', 'en': 'hip weakness'},
    'back swelling': {'ar': 'تورم الظهر', 'en': 'back swelling'},
    'ankle stiffness or tightness': {'ar': 'تيبس أو ضيق الكاحل', 'en': 'ankle stiffness or tightness'},
    'ankle weakness': {'ar': 'ضعف الكاحل', 'en': 'ankle weakness'},
    'neck weakness': {'ar': 'ضعف الرقبة', 'en': 'neck weakness'}
}


def load_models():
    """تحميل النماذج"""
    global lr_model, encoder, feature_columns

    print("\n📂 تحميل النماذج...")

    # 1. تحميل الأعمدة
    try:
        feature_columns = joblib.load('symptom_columns.pkl')
        print(f"✅ تم تحميل {len(feature_columns)} عمود من الأعراض")
    except Exception as e:
        print(f"❌ خطأ في تحميل الأعمدة: {e}")
        return False

    # 2. تحميل الـ Encoder
    try:
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        print(f"✅ تم تحميل encoder.pkl ({len(encoder.classes_)} مرض)")
    except Exception as e:
        print(f"❌ خطأ في تحميل encoder: {e}")
        return False

    # 3. تحميل Logistic Regression
    try:
        with open('logistic_regression_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        print("✅ تم تحميل logistic_regression_model.pkl")
    except Exception as e:
        print(f"❌ فشل تحميل Logistic Regression: {e}")
        return False

    return True


# تحميل النماذج عند بدء التشغيل
print("=" * 50)
print("🚀 تشغيل نظام تشخيص الأمراض")
print("=" * 50)

if load_models():
    print("\n✅ تم تحميل النماذج بنجاح!")
else:
    print("\n❌ فشل تحميل النماذج")
    exit(1)


# ==================== Routes ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': lr_model is not None,
        'encoder_loaded': encoder is not None,
        'symptoms_count': len(feature_columns) if feature_columns else 0,
        'diseases_count': len(encoder.classes_) if encoder else 0
    })


@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """قائمة الأعراض المتاحة مع الترجمة العربية"""
    symptoms = []
    for symptom_id in feature_columns:
        # تحويل underscore إلى مسافة لمطابقة القاموس
        symptom_key = symptom_id.replace('_', ' ')
        translation = SYMPTOMS_TRANSLATIONS.get(symptom_key, {})

        # إذا لم يتم العثور على الترجمة، استخدم النص الأصلي
        name_ar = translation.get('ar', symptom_key.title())
        name_en = translation.get('en', symptom_key)

        symptoms.append({
            'id': symptom_id,
            'name_ar': name_ar,
            'name_en': name_en
        })

    return jsonify({
        'success': True,
        'symptoms': symptoms,
        'total': len(feature_columns)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """التنبؤ بالمرض بناءً على الأعراض"""
    if lr_model is None:
        return jsonify({'error': 'النموذج غير متاح'}), 500

    if encoder is None:
        return jsonify({'error': 'المشفّر غير متاح'}), 500

    data = request.json
    selected_symptoms = data.get('symptoms', [])
    num_days = data.get('num_days', 1)
    language = data.get('language', 'ar')

    if not selected_symptoms:
        return jsonify({'error': 'الرجاء اختيار الأعراض'}), 400

    print(f"\n🔍 تنبؤ لـ {len(selected_symptoms)} عرض")

    # إنشاء متجه الأعراض
    input_vector = []
    for col in feature_columns:
        input_vector.append(1 if col in selected_symptoms else 0)

    input_array = np.array(input_vector).reshape(1, -1)

    # تنبؤ
    try:
        lr_pred = lr_model.predict(input_array)[0]
        disease_name = encoder.inverse_transform([lr_pred])[0]
        print(f"   ✅ النتيجة: {disease_name}")
    except Exception as e:
        print(f"   ❌ خطأ في التنبؤ: {e}")
        return jsonify({'error': f'خطأ في التنبؤ: {str(e)}'}), 500

    # حساب درجة الشدة
    severity_score = (len(selected_symptoms) * num_days) / 10
    if severity_score > 15:
        severity_level = 'critical'
    elif severity_score > 10:
        severity_level = 'high'
    elif severity_score > 5:
        severity_level = 'medium'
    else:
        severity_level = 'low'

    # توصيات طبية
    precautions = [
        " استشارة الطبيب المختص",
        " الالتزام بالعلاج الموصوف",
        " اتباع نظام غذائي صحي",
        " شرب كميات كافية من الماء",
        " الحصول على قسط كاف من الراحة"
    ]

    # وصف الحالة
    description = f"بناءً على الأعراض المختارة، قد تكون مصاباً بـ {disease_name}. هذا التشخيص استرشادي ولا يغني عن استشارة الطبيب المختص. يرجى طلب المساعدة الطبية في أقرب وقت."

    return jsonify({
        'success': True,
        'disease': disease_name,
        'description': description,
        'precautions': precautions,
        'severity_score': round(severity_score, 1),
        'severity_level': severity_level,
        'selected_symptoms_count': len(selected_symptoms),
        'num_days': num_days,
        'confidence': 85.0
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)