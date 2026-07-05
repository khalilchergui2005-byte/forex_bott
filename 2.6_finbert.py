# ============================================================
# الذرة 2.6: مراقبة الأخبار – FinBERT (تحليل المشاعر)
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("📰 تحميل نموذج FinBERT لتحليل الأخبار الاقتصادية...")
print("   (أول مرة قد تستغرق 2-3 دقائق لتحميل النموذج)")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "transformers", "torch"], capture_output=True)
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# FinBERT: نموذج مدرب خصيصاً على النصوص المالية
# نقوم بتحميله مرة واحدة
model_name = "ProsusAI/finbert"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("✅ FinBERT جاهز للتحليل")
except Exception as e:
    print(f"⚠️ فشل تحميل FinBERT: {e}")
    print("   استخدام نموذج أخف بدلاً من ذلك...")
    finbert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("✅ نموذج بديل جاهز")

# قائمة بأهم العناوين الاقتصادية المؤثرة على EUR/USD
sample_headlines = [
    "European Central Bank raises interest rates by 25 basis points",
    "US Federal Reserve signals pause in rate hikes",
    "Eurozone inflation drops unexpectedly to 2.4%",
    "US job market shows strong growth in latest NFP report",
    "Germany manufacturing PMI falls below 50 indicating contraction",
    "European Union reaches trade deal with United States",
    "Oil prices surge amid Middle East tensions weighing on EUR",
    "US dollar strengthens as Treasury yields climb higher",
    "Euro rallies on positive economic outlook from ECB President",
    "Global markets drop on recession fears hitting risk sentiment",
]

print("\n📊 تحليل عينة من العناوين الاقتصادية:")
print("=" * 70)
for headline in sample_headlines:
    result = finbert(headline)[0]
    label = result['label']
    score = result['score']
    
    # FinBERT يعطي: positive, negative, neutral
    if 'positive' in label.lower():
        impact = "🟢 إيجابي لليورو"
    elif 'negative' in label.lower():
        impact = "🔴 سلبي لليورو"
    else:
        impact = "⚪ محايد"
    
    print(f"   {impact} ({score:.2f}): {headline[:60]}...")

print("=" * 70)
print("✅ الذرة 2.6 اكتملت – FinBERT جاهز للإضافة إلى البوت الحي")
print("🎯 الخطوة التالية: دمج FinBERT مع Beast Final")
