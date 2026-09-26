# راهنمای مشترک استقرار، گزارش و نمایش عملی

برای اجرای کوتاه، پس از آماده‌شدن مدل‌ها فقط یک درخواست گفت‌وگو با هشت توکن خروجی و یک درخواست تعبیه‌سازی ارسال کنید؛ زمان هر درخواست حداکثر ۳۰ ثانیه است. این نمایش جای‌گذاری و پاسخ را نشان می‌دهد و بازپخش پنج‌دقیقه‌ای پژوهش را تکرار نمی‌کند. تعریف‌های آماده ارائه، نصب کوتاه اپراتور و نمایش سنجه‌ها در بخش‌های ۱۰ تا ۱۲ آمده‌اند.

## ۱. دامنه و یافته‌های قابل ارائه

بسته، پیکربندی‌های آزموده‌شده مدل زبانی و تعبیه‌سازی، دو مدل زبانی، بینایی و بازرتبه‌بندی، دو مدل زبانی همسان و تفکیک تانسوری و خط لوله‌ای را نگه می‌دارد. نسخه‌های اصلی بدون تغییر در پوشه configurations قرار دارند. نسخه‌های pools همان تنظیم‌های زمان اجرا را برای سه گروه گره و نسخه‌های box1 برای گره آزمایش ارائه می‌کنند. نام خدمت، انتخابگر گره و اتصال ذخیره‌سازی مطابق مقصد تعیین می‌شود. رسید منشأ، تفاوت این اتصال‌ها و سیاست اختصاصی را از تنظیم‌های آزموده‌شده مشخص می‌کند.

در اجرای نهایی تفکیک مدل، دو نمونه Qwen2.5-7B-Instruct-AWQ روی همان دو پردازنده گرافیکی قرار گرفتند. یک نمونه تفکیک تانسوری دوگانه و دیگری تفکیک خط لوله‌ای دوگانه داشت. در هر حالت اشتراکی، ۳۰۶ درخواست از ۳۰۶ درخواست موفق بود. گذردهی HAMI و HAMI + MPS به‌ترتیب حدود ۶۴٫۳۱ و ۶۴٫۳۶ توکن بر ثانیه بود. بار هر نقش ۰٫۵ درخواست بر ثانیه با ورود پواسون، به‌مدت ۳۰۰ ثانیه و سپس تخلیه صف بود. محدودیت حافظه میزبان اجازه آزمون بار اشباع‌کننده مناسب برای سنجش مزیت MPS را نداد؛ این نتیجه امکان تفکیک و اشتراک هم‌زمان را نشان می‌دهد، نه برتری قطعی یا ظرفیت بیشینه.

در ویدئو، نمودارهای ورود درخواست، صدک‌های ۹۵ و ۹۹ تأخیر و گذردهی در بازه‌های زمانی یکسان توضیح داده می‌شوند. تصویر قدیمی داشبورد صرفاً نمای رابط است. ضبط تازه با پایش فعال، یک نمایش مستقل محسوب می‌شود و نباید به اجرای فاقد پرومتئوس نسبت داده شود. مسیرهای فرمان‌های این بخش نسبت به پوشه تحویل هستند.

## ۲. پیش‌نیاز و نصب HAMI

محیط آزموده‌شده شامل دو پردازنده RTX 3080 Ti، حافظه میزبان ۱۶ گیگابایتی، کوبرنتیز سبک k3s، KServe نسخه ۰٫۱۷٫۰ و زمان اجرای vLLM نسخه ۰٫۱۵٫۱ بود. درایور، ابزار کانتینری انویدیا، کلاس اجرای nvidia و تعریف خدمت LLMInferenceService باید از قبل موجود باشند [۶]، [۷]، [۱۱]. روی گره آزمایش HAMI نصب است؛ بخش نصب تازه نباید دوباره روی آن اجرا شود. ابتدا زمینه خوشه، نسخه‌ها، منابع تخصیص‌پذیر، درخواست‌های رزروشده و وجود مدل‌ها بررسی می‌شوند.

```bash
cd /opt/GPU_Share/deliverable
kubectl config current-context
kubectl get runtimeclass nvidia
kubectl get crd llminferenceservices.serving.kserve.io
helm -n kube-system list
kubectl describe node mr-gholami
kubectl -n llm-demo get pvc
free -m
nvidia-smi
```

فرمان‌های زیر فقط برای نصب تازه HAMI نسخه ۲٫۹٫۰ هستند. نام سه گره با نام واقعی مقصد جایگزین می‌شود. هر گره باید فقط یک افزونه مسئول انتشار منابع گرافیکی داشته باشد؛ اگر افزونه پیش‌فرض انویدیا یا افزونه اپراتور فعال است، مدیریت آن باید از طریق همان نصب‌کننده به HAMI واگذار شود، بدون حذف درایور و ابزار کانتینری. فایل مقادیر، گره‌های مدیریت‌شده، سهم‌بندی دستگاه و سیاست پرکردن دستگاه را مشخص می‌کند [۵]. نصب موجود نیازمند حفظ مقادیر همان انتشار است، نه جایگزینی آن‌ها با مقادیر نمونه.

```bash
kubectl create namespace llm-demo --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace llm-demo gpu-share.io/managed=true --overwrite
kubectl label node NODE_DEDICATED NODE_HAMI NODE_MPS \
  gpu-share.io/managed=true --overwrite
helm repo add hami-charts https://project-hami.github.io/HAMi/
helm repo update hami-charts
helm install hami hami-charts/hami --version 2.9.0 \
  -n kube-system -f infrastructure/hami-values.yaml
kubectl -n kube-system get pods -o wide
```

برای گره آزمایش، سیاست موجود زمان‌بند بررسی می‌شود. سیاست دستگاه باید binpack باشد تا دو خدمت تک‌دستگاهی، در صورت کافی‌بودن سهم منابع، روی یک دستگاه قرار گیرند. سیاست spread ممکن است آن‌ها را جدا کند. تغییر سیاست در فایل مقادیر نصب موجود و ارتقای همان انتشار، در زمان نگه‌داری انجام می‌شود؛ برای نصب فعلی که این سیاست را دارد، ارتقا لازم نیست. انتخابگر گره فقط گروه را تعیین می‌کند و پردازنده مشخصی را تضمین نمی‌کند.

```bash
kubectl -n kube-system get deploy hami-scheduler \
  -o jsonpath='{range .spec.template.spec.containers[*]}{.name}{"\n"}{.command}{"\n"}{.args}{"\n"}{end}'
```

## ۳. سه گروه گره و ذخیره‌سازی

مسیرهای pools/dedicated، pools/hami و pools/hami-mps به‌ترتیب گروه اختصاصی، اشتراکی و اشتراکی با خدمت چندفرایندی را تعریف می‌کنند. در هر مسیر، 00-storage.yaml حجم پایدار محلی و ادعای متناظر را می‌سازد؛ 10-llm.yaml مدل زبانی و 11-embed.yaml مدل تعبیه‌سازی را مستقر می‌کنند. پیش از اعمال، مقدار SET در وابستگی گره حجم با نام واقعی گره، مسیر محلی با محل وزن‌ها و نشانی تصویر با مخزن قابل دسترسی مقصد جایگزین می‌شود. نشانی 127.0.0.1:5000 مخزن محلی گره آزمایش است و برای خوشه چندگرهی باید تطبیق یابد. حجم‌ها وزن مدل را دانلود نمی‌کنند.

```bash
kubectl label node NODE_DEDICATED gpu-share.io/pool=dedicated --overwrite
kubectl label node NODE_HAMI gpu-share.io/pool=hami --overwrite
kubectl label node NODE_MPS gpu-share.io/pool=hami-mps \
  gpu-share.io/mps-enabled=true --overwrite
kubectl apply --dry-run=server -f pools/hami/00-storage.yaml
kubectl apply -f pools/hami/00-storage.yaml
kubectl apply --dry-run=server -f pools/hami/10-llm.yaml
kubectl apply -f pools/hami/10-llm.yaml -f pools/hami/11-embed.yaml
```

این مثال گروه HAMI را نشان می‌دهد؛ دو مسیر دیگر با پیش‌نیاز متناظر انتخاب می‌شوند. هر حجم محلی نمونه به یک گره وابسته است. برای چند گره در یک گروه، ذخیره‌سازی مشترک یا اتصال و حجم محلی مستقل هر استقرار لازم است؛ تکثیر یک حجم محلی، داده را بین گره‌ها منتقل نمی‌کند. فایل‌ها نباید به‌صورت بازگشتی و هم‌زمان اعمال شوند.

در گروه اختصاصی، سهم محاسبات ۱۰۰ تعیین شده است تا زمان‌بند HAMI دو خدمت درخواست‌کننده سهم مثبت را روی یک دستگاه نپذیرد. این سیاست، تضمین جداسازی امنیتی فرایندهای خارج از مدیریت HAMI نیست و با خط پایه تاریخی بدون همسایه تفاوت دارد. در دو گروه اشتراکی، سهم هر خدمت ۵۰ و سهم حافظه مدل زبانی و تعبیه‌سازی به‌ترتیب ۸٬۱۹۲ و ۳٬۰۷۲ مگابایت در هر دستگاه است. سهم حافظه گرافیکی با درخواست حافظه میزبان تفاوت دارد.

## ۴. آماده‌سازی نمایش روی گره آزمایش

گره آزمایش فقط یک گره با دو پردازنده است؛ سه حالت به‌صورت متوالی نمایش داده می‌شوند. نسخه‌های box1 از ادعای حجم موجود r2-models-pvc استفاده می‌کنند و به mr-gholami محدودند. اجرای قبلی باید تمام شده و مشتری گرافیکی فعال وجود نداشته باشد. تغییر برچسب گره، پاد مقیم را تخلیه نمی‌کند؛ پیش از تغییر حالت، فقط خدمت‌های همان نمایش حذف و خروج فرایندهای آن‌ها بررسی می‌شود.

فایل زیر خدمت کنترل MPS را فقط روی گروه فعال آن محدود می‌کند و همان نام سرپرست موجود را دارد تا سرپرست دوم ساخته نشود. وصله، انتخابگر موقت باقی‌مانده از آزمون قبلی را جایگزین می‌کند. این مرحله فقط پس از تخلیه مشتریان اجرا می‌شود؛ آماده‌سازی سرپرست مسیر ارتباط مشترک را بازسازی می‌کند [۴]. روی مقصد چندگرهی نیز همین سرپرست یک بار برای کل گروه اعمال می‌شود.

```bash
kubectl label node mr-gholami gpu-share.io/pool=dedicated \
  gpu-share.io/mps-enabled=false --overwrite
kubectl apply -f infrastructure/mps-pool.yaml
kubectl -n llm-demo patch ds qwen-benchmark-mps-daemon --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/nodeSelector","value":{"gpu-share.io/pool":"hami-mps","gpu-share.io/mps-enabled":"true"}}]'
kubectl -n llm-demo get pods -l app=qwen-benchmark-mps-daemon -o wide
pgrep -af nvidia-cuda-mps || true
nvidia-smi
```

پس از پایان توقف سرپرست، هیچ فرایند کنترل یا مشتری MPS نباید باقی مانده باشد. در صورت مشاهده آن، شروع حالت اختصاصی متوقف و مالک فرایند بررسی می‌شود. سپس دو مدل اختصاصی اعمال می‌شوند. انتظار می‌رود هر خدمت روی یک دستگاه جدا قرار گیرد؛ این انتظار باید با شناسه دستگاه تخصیص‌یافته تأیید شود.

```bash
kubectl apply --dry-run=server -f box1/dedicated/10-llm.yaml \
  -f box1/dedicated/11-embed.yaml
kubectl apply -f box1/dedicated/10-llm.yaml -f box1/dedicated/11-embed.yaml
kubectl -n llm-demo get llminferenceservice,pods -o wide
```

## ۵. پاسخ واقعی و مشاهده جای‌گذاری

در هر حالت، آماده‌شدن هر دو پاد، پاسخ مدل‌ها و تخصیص دستگاه ثبت می‌شود. مقدار ARM متناسب با حالت dedicated، hami یا hami-mps انتخاب می‌شود. نشانی از درگاه پاسخ‌گوی واقعی و نام مدل از خود کارساز خوانده می‌شود. این درخواست کوتاه فقط آزمون پاسخ است و اندازه‌گیری گذردهی نیست.

```bash
ARM=dedicated
ROLE=llm
POD=$(kubectl -n llm-demo get pods \
  -l "app.kubernetes.io/name=pool-$ARM-$ROLE,kserve.io/component=workload" \
  -o jsonpath='{.items[0].metadata.name}')
kubectl -n llm-demo wait --for=condition=Ready "pod/$POD" --timeout=900s
IP=$(kubectl -n llm-demo get pod "$POD" -o jsonpath='{.status.podIP}')
BASE=
for PORT in 8000 8080 80; do
  if curl -fsS --max-time 5 "http://$IP:$PORT/v1/models" > /tmp/poc-model.json; then
    BASE="http://$IP:$PORT"; break
  fi
done
test -n "$BASE" || { echo 'No model endpoint'; exit 1; }
MODEL=$(python3 -c 'import json; print(json.load(open("/tmp/poc-model.json"))["data"][0]["id"])')
curl -fsS --max-time 30 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain GPU sharing briefly.\"}],\"max_tokens\":8}"
```

برای تعبیه‌سازی، بخش کشف پاد تا خواندن نام مدل با ROLE=embed تکرار می‌شود؛ سپس به‌جای درخواست گفت‌وگو، فرمان زیر ارسال می‌شود. پاسخ هر دو مدل باید ضبط شود، نه صرفاً وضعیت آماده.

```bash
curl -fsS --max-time 30 "$BASE/v1/embeddings" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"input\":[\"GPU sharing demonstration\"]}"
```

فرمان‌های زیر تخصیص دستگاه و فرایندهای مقیم را نشان می‌دهند. در اشتراک تک‌دستگاهی، دو پاد باید شناسه مشترک داشته باشند؛ در تفکیک، هر نقش باید روی هر دو دستگاه مشاهده شود. شناسه تخصیص‌یافته با فرایندهای واقعی تطبیق داده می‌شود.

```bash
kubectl -n llm-demo get pods -o \
  'custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,GPU:.metadata.annotations.hami\.io/vgpu-devices-allocated'
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
nvidia-smi pmon -c 1
```

## ۶. تغییر از اختصاصی به اشتراک HAMI

دو خدمت اختصاصی حذف می‌شوند. پس از حذف پادهای آن‌ها و خالی‌شدن دستگاه‌ها، گروه به hami تغییر می‌کند و فایل‌های اشتراکی اعمال می‌شوند. MPS خاموش می‌ماند. آزمون پاسخ و جای‌گذاری بخش قبل با ARM=hami تکرار می‌شود. سهم‌بندی HAMI به معنی استفاده از MPS نیست و حضور دو پاد نیز به‌تنهایی اشتراک یک دستگاه را ثابت نمی‌کند [۵].

```bash
kubectl delete -f box1/dedicated/10-llm.yaml -f box1/dedicated/11-embed.yaml
kubectl -n llm-demo get pods -o wide
nvidia-smi
```

پس از تأیید خروج مشتریان، مرحله بعد اجرا می‌شود.

```bash
kubectl label node mr-gholami gpu-share.io/pool=hami --overwrite
kubectl apply -f box1/hami/10-llm.yaml -f box1/hami/11-embed.yaml
kubectl -n llm-demo get llminferenceservice,pods -o wide
```

## ۷. فعال‌سازی و توقف MPS

پس از نمایش HAMI، دو خدمت آن حذف و خروج مشتریان تأیید می‌شود. سپس گروه hami-mps و برچسب فعال‌سازی انتخاب می‌شوند. سرپرست، خدمت کنترل چندفرایندی را راه‌اندازی می‌کند. فایل‌های مدل، مسیر مشترک ارتباط و تنظیم سقف رشته فعال دارند. سقف رشته فعال با سهم سخت پردازنده فیزیکی برابر نیست؛ تغییر آن برای مشتری جدید اعمال می‌شود و جای تأیید اتصال واقعی را نمی‌گیرد [۴].

```bash
kubectl delete -f box1/hami/10-llm.yaml -f box1/hami/11-embed.yaml
kubectl -n llm-demo get pods -o wide
nvidia-smi
```

پس از خروج مشتریان، سرپرست و سپس مدل‌ها فعال می‌شوند.

```bash
kubectl label node mr-gholami gpu-share.io/pool=hami-mps \
  gpu-share.io/mps-enabled=true --overwrite
kubectl -n llm-demo rollout status ds/qwen-benchmark-mps-daemon --timeout=180s
kubectl apply -f box1/hami-mps/10-llm.yaml -f box1/hami-mps/11-embed.yaml
kubectl -n llm-demo get llminferenceservice,pods -o wide
echo get_server_list | env CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
  nvidia-cuda-mps-control
nvidia-smi pmon -c 1
```

آزمون پاسخ با ARM=hami-mps تکرار می‌شود. پس از پاسخ، فهرست کارساز نباید خالی باشد. شناسه هر کارساز خروجی فرمان قبل در فرمان زیر قرار می‌گیرد و مشتریان با پادهای مدل تطبیق داده می‌شوند. در تفکیک مشترک، چهار کارگر مدل انتظار می‌رود. نام فرایند یا وجود سوکت بدون مشتری، اثبات مسیر MPS نیست. نسخه آزموده‌شده از شناسه کاربری یکسان استفاده می‌کند؛ اشتراک کاربران با شناسه متفاوت در این پیکربندی تضمین نشده است.

```bash
echo "get_client_list SERVER_PID" | env CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
  nvidia-cuda-mps-control
```

برای توقف، ابتدا خدمت‌های متصل حذف می‌شوند؛ پس از خروج مشتریان، برچسب فعال‌سازی false می‌شود. حذف متغیرهای محیطی مدل به‌تنهایی خدمت کنترل میزبان را متوقف نمی‌کند.

```bash
kubectl delete -f box1/hami-mps/10-llm.yaml -f box1/hami-mps/11-embed.yaml
kubectl -n llm-demo get pods -o wide
nvidia-smi
```

پس از تأیید تخلیه، خاموش‌سازی انجام می‌شود.

```bash
kubectl label node mr-gholami gpu-share.io/mps-enabled=false --overwrite
kubectl -n llm-demo get pods -l app=qwen-benchmark-mps-daemon -o wide
pgrep -af nvidia-cuda-mps || true
```

## ۸. تفکیک مدل و محدودیت حافظه

فایل‌های split/tp.yaml و split/pp.yaml زیر هر گروه اشتراکی، آرگومان‌های اجرای موفق را حفظ کرده‌اند. درجه تانسوری و خط لوله‌ای در نقش نخست دو و یک و در نقش دوم یک و دو است. هر پاد دو دستگاه، سهم حافظه ۵٬۶۳۲ مگابایت در هر دستگاه و حافظه میزبان درخواستی ۴ گیگابایت با حد ۶ گیگابایت دارد. این دو فایل جایگزین جفت مدل زبانی و تعبیه‌سازی‌اند، نه خدمت اضافه کنار آن‌ها. ابتدا جفت قبلی حذف و مرحله روشن یا خاموش‌کردن MPS متناسب با حالت انجام می‌شود.

به‌دلیل کمبود حافظه میزبان، طول زمینه ۱٬۰۲۴، دنباله هم‌زمان دو، توکن دسته پیش‌پردازش ۲۵۶، حافظه نهان کلید و مقدار ۲۵۶ مگابایت در هر دستگاه، اجرای مشتاق، بارگذاری تنبل وزن، روش ساخت کارگر فورک و غیرفعال‌بودن حافظه نهان پیشوند انتخاب شدند. کوچک‌کردن وزن‌ها به‌تنهایی سربار فرایندها و کتابخانه‌ها را حذف نکرد. در اجرای ثبت‌شده، پرومتئوس، صادرکننده‌های سنجه و رابط پایش موقتاً متوقف شدند تا حافظه واقعی یا رزرو زمان‌بند آزاد شود.

برای نمایش دوباره با پایش فعال، موفقیت قبلی کافی نیست. جمع درخواست‌های دو پاد باید در ظرفیت تخصیص‌پذیر باقی‌مانده جا شود و حافظه واقعی برای اوج راه‌اندازی و بار هر دو نقش، به‌اضافه دست‌کم یک گیگابایت ذخیره، دوباره اندازه‌گیری شود. تا تأیید این شرط، فقط نتایج ثبت‌شده نمایش داده می‌شوند و جفت تفکیک‌شده شروع نمی‌شود. فرمان زیر برای حالت HAMI و فقط پس از پذیرش منابع است.

```bash
free -m
kubectl describe node mr-gholami
kubectl label node mr-gholami gpu-share.io/pool=hami \
  gpu-share.io/mps-enabled=false --overwrite
kubectl apply --dry-run=server -f box1/hami/split/tp.yaml \
  -f box1/hami/split/pp.yaml
kubectl apply -f box1/hami/split/tp.yaml -f box1/hami/split/pp.yaml
kubectl -n llm-demo get llminferenceservice,pods -o wide
```

برای حالت ترکیبی، فایل‌ها از box1/hami-mps/split پس از تخلیه نقش‌های قبلی و روشن‌کردن سرپرست انتخاب می‌شوند. درگاه پاسخ نقش‌های تفکیک‌شده ۸۰۰۰ است؛ در آزمون پاسخ، نام خدمت و نقش به tp یا pp تغییر می‌کند. پس از نمایش فقط فایل‌های همان جفت حذف می‌شوند. خط پایه تاریخی در configurations/split-tp2-pp2 برای هر نقش جداگانه اجرا می‌شود.

با حافظه کافی، پایش حفظ و اوج راه‌اندازی مشترک اندازه‌گیری می‌شود. روش ساخت کارگر از فورک به اسپاون و سپس طول زمینه و اندازه دسته، هر بار یک متغیر، بازآزمایی می‌شوند. درخواست و حد حافظه میزبان از اندازه‌گیری تازه به دست می‌آید؛ کاهش درخواست، مصرف واقعی را کاهش نمی‌دهد. بررسی مزیت MPS نیز به بار مناسب و خط پایه و بازپخش برابر تازه نیاز دارد.

## ۹. پایش، ضبط و پذیرش مقصد

فایل monitoring/servicemonitors.yaml هدف گردآوری سنجه را به پرومتئوس موجود معرفی می‌کند و monitoring/dashboard.yaml داشبورد گرافانای موجود را تعریف می‌کند؛ این دو فایل سامانه پایش را نصب نمی‌کنند. سپس monitoring/pool-workloads.yaml نگاشت نام خدمت را برای نسخه‌های گروه گره اصلاح می‌کند؛ همان هدف موجود به‌روزرسانی می‌شود تا گردآوری تکراری رخ ندهد. نام گروه و نقش در برچسب سنجه حفظ می‌شود. انتخابگرهای فضای نام، برچسب و هدف شبکه باید با نصب مقصد تطبیق یابند. در گره آزمایش، بازگرداندن پایش باید از تنظیمات پشتیبان همان نصب انجام شود. حضور اجزای تاریخی Knative در خوشه، به معنی استفاده این خدمت‌ها از همه قابلیت‌های آن نیست [۱۲].

```bash
kubectl apply --dry-run=server -f monitoring/servicemonitors.yaml \
  -f monitoring/dashboard.yaml
kubectl apply -f monitoring/servicemonitors.yaml -f monitoring/dashboard.yaml
kubectl apply -f monitoring/pool-workloads.yaml
kubectl -n llm-demo get llminferenceservice,pods -o wide
```

در نمایش، نام حالت، بازه زمانی جاری، جای‌گذاری واقعی و پاسخ مدل‌ها نشان داده می‌شود. نمودارهای پژوهش از نتایج ذخیره‌شده خوانده می‌شوند؛ چند درخواست کوتاه آزمون ظرفیت نیستند. شواهد اجرای مدل از آزمایش‌های ثبت‌شده و پیکربندی‌های اصلی همراه بسته به دست آمده است. بررسی‌های عملیاتی شامل پاسخ هر دو مدل، نبود بازراه‌اندازی، تطبیق شناسه دستگاه‌ها، وضعیت درست MPS و پایش فعال است. نام گره، مخزن تصویر، مسیر وزن و ظرفیت تخصیص‌پذیر، مقادیر وابسته به محیط مقصد هستند [۱۳]، [۱۴].

## ۱۰. تعریف کوتاه برای ارائه

HAMI لایه مدیریت و اشتراک پردازنده گرافیکی در کوبرنتیز است که با زمان‌بند، افزونه دستگاه و کنترل نرم‌افزاری فراخوانی‌های CUDA، چند پاد را بر اساس سهم حافظه و محاسبات روی دستگاه فیزیکی جای می‌دهد؛ در پروژه، تخصیص با شناسه واقعی دستگاه بررسی شد. صفحه قابل نمایش: https://project-hami.io/docs/core-concepts/gpu-virtualization

MPS خدمت چندفرایندی انویدیا است که امکان اجرای هم‌زمان کارهای CUDA از چند فرایند را روی یک پردازنده فراهم می‌کند؛ یافته‌های پروژه نشان می‌دهند اثر آن به بار و ترکیب مستأجران وابسته است، بنابراین گذردهی کل باید همراه تأخیر هر مدل تفسیر شود. صفحه قابل نمایش: https://docs.nvidia.com/deploy/mps/when-to-use-mps.html

در اجرای تفکیک‌شده، محدودیت حافظه میزبان مانع بار اشباع‌کننده مناسب شد؛ نتیجه آن اجرا، امکان اشتراک همراه تفکیک بود. MPS جای زمان‌بند HAMI را نمی‌گیرد و HAMI نیز به‌تنهایی اتصال مشتری به MPS را ثابت نمی‌کند. کنترل مشتریان در بخش ۷ آمده است.

## ۱۱. اپراتور انویدیا و نصب کوتاه

اپراتور پردازنده گرافیکی انویدیا، اجزای زیرساختی مانند درایور، ابزار کانتینری و صادرکننده سنجه را در کوبرنتیز مدیریت می‌کند؛ در ترکیب این پروژه، مسئولیت افزونه تخصیص دستگاه با HAMI است. راهنمای رسمی نصب: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.3/getting-started.html

روی Box1 اپراتور از قبل نصب است؛ فقط وضعیت آن را نشان دهید و نصب را تکرار نکنید. فرمان نصب زیر برای مقصد تازه‌ای است که مانند محیط پروژه، درایور و ابزار کانتینری انویدیا از قبل نصب و زمان اجرای nvidia پیکربندی شده‌اند. نسخه ۲۵٫۳٫۰ همان نسخه ثبت‌شده محیط است؛ این فرمان بازسازی تنظیم نصب برای همین پیش‌نیازهاست. با غیرفعال‌کردن افزونه دستگاه اپراتور، دو افزونه رقیب برای منبع گرافیکی ایجاد نمی‌شود. اگر کشف ویژگی گره از قبل نصب است، مقدار nfd.enabled=false نیز افزوده شود.

```bash
helm -n gpu-operator list
kubectl get clusterpolicy
kubectl -n gpu-operator get pods -o wide
```

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update nvidia
helm install gpu-operator nvidia/gpu-operator --version v25.3.0 \
  -n gpu-operator --create-namespace --wait \
  --set driver.enabled=false --set toolkit.enabled=false \
  --set devicePlugin.enabled=false --set migManager.enabled=false
```

این نمونه نصب، درایور یا ابزار کانتینری را نصب نمی‌کند؛ روی میزبان فاقد آن‌ها ابتدا همان پیش‌نیازها برقرار شوند. در نصب فعلی Box1، تغییر تنظیم اپراتور فقط برای فیلم‌برداری لازم نیست. سپس مراحل نصب HAMI در بخش ۲ و فعال‌سازی MPS در بخش ۷ دنبال می‌شوند.

## ۱۲. نمایش کوتاه جای‌گذاری، سنجه و داشبورد

روی خود گره گرافیکی، فرمان زیر برای ۲۰ ثانیه وضعیت دستگاه‌ها را نشان می‌دهد. هم‌زمان، یک درخواست بخش ۵ ارسال کنید؛ سپس شناسه تخصیص پاد، شناسه دستگاه و فرایند مدل را کنار هم نمایش دهید. پایان زمان فرمان timeout طبیعی است. بالا رفتن میزان استفاده از دستگاه به‌تنهایی اثبات هم‌مکانی دو مدل نیست.

```bash
timeout 20s nvidia-smi --loop=1
kubectl -n llm-demo get pods -o \
  'custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,GPU:.metadata.annotations.hami\.io/vgpu-devices-allocated'
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

زنجیره پایش چنین است: نقطه /metrics زمان اجرا و صادرکننده‌های HAMI و انویدیا، هدف‌های PodMonitor و ServiceMonitor، پرومتئوس، و سپس گرافانا یا رابط HAMI. برای نشان‌دادن سهم تنظیم‌شده، بخش resources فایل مدل و حاشیه‌نویسی تخصیص را باز کنید؛ برای مصرف و عملکرد واقعی، سنجه‌های گردآوری‌شده را نشان دهید. بخش ۹ فایل‌های هدف پایش را اعمال می‌کند. روی Box1، نشانی محلی زیر از تنظیم ثبت‌شده زمان‌بند است؛ متغیر BASE از کشف نشانی مدل در بخش ۵ می‌آید.

```bash
curl -fsS --max-time 5 http://127.0.0.1:31993/metrics \
  | grep -E '^hami_gpu_(core|memory)' | head -n 12
curl -fsS --max-time 5 "$BASE/metrics" \
  | grep -E '^vllm:(num_requests_running|num_requests_waiting|generation_tokens_total)' \
  | head -n 12
kubectl get servicemonitor,podmonitor -A
```

برای مرورگر، انتقال درگاه را روی همان رایانه‌ای اجرا کنید که مرورگر و دسترسی kubectl به خوشه دارد؛ این فرمان به SSH نیاز ندارد. هر انتقال درگاه در یک پایانه جدا باز می‌ماند و با Ctrl+C پایان می‌یابد. سامانه پایش و رابط HAMI باید قبلاً از حالت توقف برگردانده شده باشند. اگر خدمت موجود نیست یا آماده نیست، انتقال درگاه آن را نصب یا فعال نمی‌کند.

```bash
kubectl -n default port-forward svc/prometheus-operated 9090:9090
```

در مرورگر، http://127.0.0.1:9090/targets را باز کنید؛ هدف زمان اجرای مدل و HAMI باید فعال باشند. برای گرافانا، نام خدمت با برچسب رسمی کشف می‌شود؛ اگر نصب شما در فضای نام دیگری است، default را جایگزین کنید.

```bash
GRAFANA_SVC=$(kubectl -n default get svc \
  -l app.kubernetes.io/name=grafana -o jsonpath='{.items[0].metadata.name}')
GRAFANA_PORT=$(kubectl -n default get svc "$GRAFANA_SVC" \
  -o jsonpath='{.spec.ports[0].port}')
kubectl -n default port-forward "svc/$GRAFANA_SVC" "3000:$GRAFANA_PORT"
```

در مرورگر، http://127.0.0.1:3000/d/r2-gpu-sharing را باز کنید. برای رابط HAMI، از فهرست زیر خدمت رابط وب را انتخاب کنید، نه خدمت صادرکننده dcgm؛ سپس درگاه وب آن را از خروجی مشخص کنید. نام و درگاه از نصب واقعی خوانده می‌شوند تا به نام حدسی متکی نباشند.

```bash
kubectl -n kube-system get svc -l app.kubernetes.io/instance=my-hami-webui
read -r -p 'HAMi WebUI service name: ' HAMI_WEB_SVC
kubectl -n kube-system get svc "$HAMI_WEB_SVC" -o wide
read -r -p 'Web service port: ' HAMI_WEB_PORT
kubectl -n kube-system port-forward "svc/$HAMI_WEB_SVC" "3001:$HAMI_WEB_PORT"
```

صفحه http://127.0.0.1:3001 را برای نمایش گره، دستگاه و سهم پادها باز کنید. سنجه‌های زیر در بخش جست‌وجوی پرومتئوس یا کاوش گرافانا قابل نمایش‌اند. دو درخواست کوتاه برای نتیجه‌گیری درباره صدک‌ها کافی نیستند؛ نمودارهای تأخیر پژوهش از اجرای ثبت‌شده خوانده می‌شوند. نبود سری سنجه با مقدار صفر یکسان نیست؛ ابتدا فعال‌بودن هدف گردآوری و وجود نام سنجه بررسی شود.

```promql
hami_gpu_memory_allocated_bytes
hami_gpu_core_allocated_ratio
sum by (r2_role) (vllm:num_requests_running{namespace="llm-demo"})
sum by (r2_role) (rate(vllm:generation_tokens_total{namespace="llm-demo"}[1m]))
```

بهترین ضبط کوتاه، یک پاسخ گفت‌وگو، یک پاسخ تعبیه‌سازی و حدود ۲۰ تا ۳۰ ثانیه نمایش جای‌گذاری و داشبورد است؛ دانلود وزن و آماده‌سازی اولیه مدل جزو زمان این آزمون کوتاه نیست.
