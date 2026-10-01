# راهنمای مشترک استقرار، گزارش و نمایش عملی

## مسیر ضبط روی Box1

تمام فرمان‌های این راهنما برای Bash هستند. بلوک چندخطی، تابع، حلقه و شرط را کامل کپی کنید؛ خط‌های مستقل را می‌توان جدا اجرا کرد. همه مسیرها از `/opt/GPU_Share/deliverable` هستند. روایت ویدئو در راهنمای محلی VIDEO_SCENARIO_FA است؛ فرمان‌ها فقط اینجا نگه داشته می‌شوند. فایل گزارش پژوهش در این مرحله تغییر نمی‌کند.

ترتیب ضبط: ابتدا بررسی زیر و بخش‌های ۹ و ۱۰ برای پایش؛ سپس معرفی YAMLهای بخش ۳؛ بعد بخش‌های ۴ و ۵ برای dedicated، بخش ۶ برای HAMi و بخش ۷ برای HAMi+MPS. در انتها فقط بلوک پاک‌سازی بخش ۷ اجرا شود. بخش ۸ معرفی نتیجه ثبت‌شده split است و در این نمایش کوتاه، با پایش روشن دوباره اجرا نمی‌شود. سه مقدار `dedicated`، `hami` و `hami-mps` برای `gpu-share.io/pool` حفظ می‌شوند؛ یک گره در هر لحظه فقط یکی از آن‌ها را دارد. بلوک تولیدی سه‌گرهی بخش ۳ روی Box1 اجرا نشود.

```bash
cd /opt/GPU_Share/deliverable
kubectl get node mr-gholami -L gpu-share.io/pool,gpu-share.io/mps-enabled
helm -n kube-system status hami
helm -n default status prometheus
kubectl -n default get pods,svc -o wide
kubectl -n kube-system get deployment my-hami-webui
kubectl -n gpu-operator get ds nvidia-dcgm-exporter
kubectl -n llm-demo get pvc r2-models-pvc
kubectl -n llm-demo get llminferenceservice,pods -o wide
free -m
nvidia-smi
```

اگر پایش آماده است، نصب و بازیابی بخش ۹ را رد کنید و فقط بررسی هدف‌ها و اصلاح سازگاری سنجه در بخش ۱۲ را انجام دهید. `Running` به‌تنهایی کافی نیست؛ پاد باید Ready، خدمت دارای Endpoint و هدف پرومتئوس UP باشد. موفقیت آزمون HTTP نیز با HTTP 200 و بدنه پاسخ معتبر تعیین می‌شود. اگر هر مرحله خطا داد، به حالت بعد نروید. این راهنما وضعیت فعلی سرور را بدون دریافت خروجی آن تأیید نمی‌کند.

برای اجرای کوتاه، پس از آماده‌شدن مدل‌ها فقط یک درخواست گفت‌وگو با هشت توکن خروجی و یک درخواست تعبیه‌سازی ارسال کنید؛ زمان هر درخواست حداکثر ۳۰ ثانیه است. این نمایش جای‌گذاری و پاسخ را نشان می‌دهد و بازپخش پنج‌دقیقه‌ای پژوهش را تکرار نمی‌کند. نصب HAMI، سه حالت اجرای اختصاصی، HAMI و HAMI + MPS، بازگرداندن پایش و نمایش داشبوردها در بخش‌های بعد آمده‌اند.

## ۱. دامنه و یافته‌های قابل ارائه

بسته، پیکربندی‌های آزموده‌شده مدل زبانی و تعبیه‌سازی، دو مدل زبانی، بینایی و بازرتبه‌بندی، دو مدل زبانی همسان و تفکیک تانسوری و خط لوله‌ای را نگه می‌دارد. نسخه‌های اصلی بدون تغییر در پوشه configurations قرار دارند. نسخه‌های pools همان تنظیم‌های زمان اجرا را برای سه گروه گره و نسخه‌های box1 برای گره آزمایش ارائه می‌کنند. نام خدمت، انتخابگر گره و اتصال ذخیره‌سازی مطابق مقصد تعیین می‌شود. رسید منشأ، تفاوت این اتصال‌ها و سیاست اختصاصی را از تنظیم‌های آزموده‌شده مشخص می‌کند.

در اجرای نهایی تفکیک مدل، دو نمونه Qwen2.5-7B-Instruct-AWQ روی همان دو پردازنده گرافیکی قرار گرفتند. یک نمونه تفکیک تانسوری دوگانه و دیگری تفکیک خط لوله‌ای دوگانه داشت. در هر حالت اشتراکی، ۳۰۶ درخواست از ۳۰۶ درخواست موفق بود. گذردهی HAMI و HAMI + MPS به‌ترتیب حدود ۶۴٫۳۱ و ۶۴٫۳۶ توکن بر ثانیه بود. بار هر نقش ۰٫۵ درخواست بر ثانیه با ورود پواسون، به‌مدت ۳۰۰ ثانیه و سپس تخلیه صف بود. محدودیت حافظه میزبان اجازه آزمون بار اشباع‌کننده مناسب برای سنجش مزیت MPS را نداد؛ این نتیجه امکان تفکیک و اشتراک هم‌زمان را نشان می‌دهد، نه برتری قطعی یا ظرفیت بیشینه.

در ویدئو، نمودارهای ورود درخواست، صدک‌های ۹۵ و ۹۹ تأخیر و گذردهی در بازه‌های زمانی یکسان توضیح داده می‌شوند. تصویر قدیمی داشبورد صرفاً نمای رابط است. ضبط تازه با پایش فعال، یک نمایش مستقل محسوب می‌شود و نباید به اجرای فاقد پرومتئوس نسبت داده شود. مسیرهای فرمان‌های این بخش نسبت به پوشه تحویل هستند.

## ۲. پیش‌نیاز و نصب HAMI

محیط آزموده‌شده شامل دو پردازنده RTX 3080 Ti، حافظه میزبان ۱۶ گیگابایتی، کوبرنتیز سبک k3s، KServe نسخه ۰٫۱۷٫۰ و زمان اجرای vLLM نسخه ۰٫۱۵٫۱ بود. درایور، ابزار کانتینری انویدیا، کلاس اجرای nvidia و تعریف خدمت LLMInferenceService باید از قبل موجود باشند [۶]، [۷]، [۱۱]. روی گره آزمایش HAMI نصب است؛ در ویدئوی همین گره، نصب موجود فقط شناسایی و بررسی می‌شود. روی یک گره GPU تازه، همان بلوک در نبود انتشار HAMI نصب نسخه ۲٫۹٫۰ را انجام می‌دهد. ابتدا زمینه خوشه، نسخه‌ها، منابع تخصیص‌پذیر، درخواست‌های رزروشده و وجود مدل‌ها بررسی می‌شوند.

```bash
cd /opt/GPU_Share/deliverable
kubectl config current-context
kubectl get nodes -o wide
kubectl get runtimeclass nvidia
kubectl get crd llminferenceservices.serving.kserve.io
helm -n kube-system list
free -m
nvidia-smi
```

نام واقعی Box1 در بلوک زیر mr-gholami است. روی مقصد تازه فقط نام واقعی گره جایگزین می‌شود؛ روی Box1 همین مقدار حفظ شود. هر گره باید فقط یک افزونه مسئول انتشار منابع گرافیکی داشته باشد؛ اگر افزونه پیش‌فرض انویدیا یا افزونه اپراتور فعال است، مدیریت آن از طریق همان نصب‌کننده به HAMi واگذار می‌شود، بدون حذف driver و toolkit. فایل مقادیر، گره‌های مدیریت‌شده و سیاست جای‌گذاری را مشخص می‌کند. نصب موجود دوباره با `helm install` ساخته نمی‌شود و مقادیر همان انتشار حفظ می‌شوند.

```bash
GPU_NODE=mr-gholami
echo "GPU node: $GPU_NODE"

kubectl create namespace llm-demo --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace llm-demo gpu-share.io/managed=true --overwrite
kubectl label node "$GPU_NODE" gpu=on gpu-share.io/managed=true --overwrite

helm repo add hami-charts https://project-hami.github.io/HAMi/ --force-update
helm repo update hami-charts

if helm -n kube-system status hami >/dev/null 2>&1; then
  echo 'HAMI is already installed; keeping the existing release.'
  helm -n kube-system status hami
else
  helm install hami hami-charts/hami --version 2.9.0 \
    -n kube-system -f infrastructure/hami-values.yaml
fi

kubectl -n kube-system get pods -o wide
kubectl get node "$GPU_NODE" -L gpu,gpu-share.io/managed,gpu-share.io/pool,gpu-share.io/mps-enabled
```

برای گره آزمایش، سیاست موجود زمان‌بند بررسی می‌شود. سیاست دستگاه باید binpack باشد تا دو خدمت تک‌دستگاهی، در صورت کافی‌بودن سهم منابع، روی یک دستگاه قرار گیرند. سیاست spread ممکن است آن‌ها را جدا کند. تغییر سیاست در فایل مقادیر نصب موجود و ارتقای همان انتشار، در زمان نگه‌داری انجام می‌شود؛ برای نصب فعلی که این سیاست را دارد، ارتقا لازم نیست. انتخابگر گره فقط گروه را تعیین می‌کند و پردازنده مشخصی را تضمین نمی‌کند.

```bash
kubectl -n kube-system get deploy hami-scheduler \
  -o jsonpath='{range .spec.template.spec.containers[*]}{.name}{"\n"}{.command}{"\n"}{.args}{"\n"}{end}'
```

## ۳. سه گروه گره و ذخیره‌سازی

مسیرهای pools/dedicated، pools/hami و pools/hami-mps به‌ترتیب گروه اختصاصی، اشتراکی و اشتراکی با خدمت چندفرایندی را تعریف می‌کنند. در هر مسیر، 00-storage.yaml حجم پایدار محلی و ادعای متناظر را می‌سازد؛ 10-llm.yaml مدل زبانی و 11-embed.yaml مدل تعبیه‌سازی را مستقر می‌کنند. پیش از اعمال، مقدار SET در وابستگی گره حجم با نام واقعی گره، مسیر محلی با محل وزن‌ها و نشانی تصویر با مخزن قابل دسترسی مقصد جایگزین می‌شود. نشانی 127.0.0.1:5000 مخزن محلی گره آزمایش است و برای خوشه چندگرهی باید تطبیق یابد. حجم‌ها وزن مدل را دانلود نمی‌کنند.

در استقرار تولیدی سه‌گرهی، هر گروه روی گره فیزیکی جدا برچسب می‌خورد. این بلوک نمونه فقط پس از جایگزینی سه نام با گره‌های واقعی اجرا می‌شود. در Box1 سه گره وجود ندارد و همین سه حالت طبق بخش‌های بعدی روی `mr-gholami` به‌صورت متوالی نمایش داده می‌شوند.

```bash
# Production only: set these three variables to real node names.
# Leave them empty on Box1; sections 4-7 switch the single node sequentially.
NODE_DEDICATED=${NODE_DEDICATED:-}
NODE_HAMI=${NODE_HAMI:-}
NODE_MPS=${NODE_MPS:-}

if [ -n "$NODE_DEDICATED" ] && [ -n "$NODE_HAMI" ] && [ -n "$NODE_MPS" ]; then
  kubectl label node "$NODE_DEDICATED" gpu=on gpu-share.io/managed=true \
    gpu-share.io/pool=dedicated gpu-share.io/mps-enabled=false --overwrite
  kubectl label node "$NODE_HAMI" gpu=on gpu-share.io/managed=true \
    gpu-share.io/pool=hami gpu-share.io/mps-enabled=false --overwrite
  kubectl label node "$NODE_MPS" gpu=on gpu-share.io/managed=true \
    gpu-share.io/pool=hami-mps gpu-share.io/mps-enabled=true --overwrite
else
  echo 'Production node names are not set; skipping three-node labeling and using sequential Box1 modes.'
fi

kubectl get nodes -L gpu,gpu-share.io/managed,gpu-share.io/pool,gpu-share.io/mps-enabled
```

این مثال گروه HAMI را برای ذخیره‌سازی و خدمت‌های مقصد نشان می‌دهد؛ دو مسیر دیگر با پیش‌نیاز متناظر انتخاب می‌شوند. هر حجم محلی نمونه به یک گره وابسته است. برای چند گره در یک گروه، ذخیره‌سازی مشترک یا اتصال و حجم محلی مستقل هر استقرار لازم است؛ تکثیر یک حجم محلی، داده را بین گره‌ها منتقل نمی‌کند. فایل‌ها نباید به‌صورت بازگشتی و هم‌زمان اعمال شوند.

```bash
kubectl apply --dry-run=server -f pools/hami/00-storage.yaml
kubectl apply -f pools/hami/00-storage.yaml
kubectl apply --dry-run=server -f pools/hami/10-llm.yaml
kubectl apply -f pools/hami/10-llm.yaml -f pools/hami/11-embed.yaml
```

در گروه اختصاصی، سهم محاسبات ۱۰۰ تعیین شده است تا زمان‌بند HAMI دو خدمت درخواست‌کننده سهم مثبت را روی یک دستگاه نپذیرد. این سیاست، تضمین جداسازی امنیتی فرایندهای خارج از مدیریت HAMI نیست و با خط پایه تاریخی بدون همسایه تفاوت دارد. در دو گروه اشتراکی، سهم هر خدمت ۵۰ و سهم حافظه مدل زبانی و تعبیه‌سازی به‌ترتیب ۸٬۱۹۲ و ۳٬۰۷۲ مگابایت در هر دستگاه است. سهم حافظه گرافیکی با درخواست حافظه میزبان تفاوت دارد.

## ۴. آماده‌سازی نمایش روی گره آزمایش

گره آزمایش فقط یک گره با دو پردازنده است؛ سه حالت به‌صورت متوالی نمایش داده می‌شوند. نسخه‌های box1 از ادعای حجم موجود r2-models-pvc استفاده می‌کنند و به mr-gholami محدودند. اجرای قبلی باید تمام شده و مشتری گرافیکی فعال وجود نداشته باشد. تغییر برچسب گره، پاد مقیم را تخلیه نمی‌کند؛ پیش از تغییر حالت، فقط خدمت‌های همان نمایش حذف و خروج فرایندهای آن‌ها بررسی می‌شود.

فایل زیر خدمت کنترل MPS را فقط روی گروه فعال آن محدود می‌کند و همان نام سرپرست موجود را دارد تا سرپرست دوم ساخته نشود. این مرحله فقط پس از تخلیه مشتریان اجرا می‌شود؛ آماده‌سازی سرپرست مسیر ارتباط مشترک را بازسازی می‌کند [۴]. روی مقصد چندگرهی نیز همین سرپرست یک بار برای کل گروه اعمال می‌شود.

```bash
GPU_NODE=mr-gholami

kubectl delete -f box1/hami-mps/10-llm.yaml -f box1/hami-mps/11-embed.yaml --ignore-not-found=true
kubectl delete -f box1/hami/10-llm.yaml -f box1/hami/11-embed.yaml --ignore-not-found=true
kubectl delete -f box1/dedicated/10-llm.yaml -f box1/dedicated/11-embed.yaml --ignore-not-found=true

kubectl -n llm-demo wait --for=delete pod \
  -l 'app.kubernetes.io/name in (pool-dedicated-llm,pool-dedicated-embed,pool-hami-llm,pool-hami-embed,pool-hami-mps-llm,pool-hami-mps-embed)' --timeout=180s

kubectl label node "$GPU_NODE" gpu-share.io/pool=dedicated \
  gpu-share.io/mps-enabled=false --overwrite
kubectl apply -f infrastructure/mps-pool.yaml
kubectl -n llm-demo get pods -l app=qwen-benchmark-mps-daemon -o wide
pgrep -af nvidia-cuda-mps || true
nvidia-smi
```

پس از پایان توقف سرپرست، daemon و مشتری MPS نباید باقی مانده باشند. `pgrep -af` ممکن است خود wrapper را نیز پیدا کند؛ طبق بررسی harness، پاسخ control socket معیار دقیق‌تری است. بلوک زیر روی خود Box1 با دسترسی همان کاربر daemon، در این نصب root، اجرا می‌شود. نبود دسترسی نباید خاموش‌بودن تفسیر شود. تا وقتی پیام STOP دیده می‌شود، مدل‌های dedicated یا HAMi شروع نشوند. حذف متغیر محیطی MPS به‌تنهایی برای خاموش‌کردن مسیر کافی نیست.

```bash
if [ "$(id -u)" -ne 0 ]; then
  echo 'STOP: perform the host MPS check from the root terminal used for this daemon.'
elif [ -S /tmp/nvidia-mps/control ] && \
  echo get_server_list | timeout 10 env CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
    nvidia-cuda-mps-control; then
  echo 'STOP: MPS control is still reachable.'
else
  echo 'MPS control is not reachable; also confirm its pod and GPU clients are gone.'
fi
```

سپس دو مدل dedicated اعمال می‌شوند. انتظار می‌رود هر خدمت روی یک دستگاه جدا قرار گیرد؛ این انتظار با UUID تأیید می‌شود. در انتقال بعدی به HAMi نیز همین بررسی خاموش‌بودن تکرار شود.

```bash
kubectl apply --dry-run=server -f box1/dedicated/10-llm.yaml \
  -f box1/dedicated/11-embed.yaml
kubectl apply -f box1/dedicated/10-llm.yaml -f box1/dedicated/11-embed.yaml
kubectl -n llm-demo get llminferenceservice,pods -o wide
```

## ۵. پاسخ واقعی و مشاهده جای‌گذاری

در هر حالت، آماده‌شدن هر دو پاد، پاسخ مدل‌ها و تخصیص دستگاه ثبت می‌شود. برای جلوگیری از وابستگی به مسیر شبکه Pod IP، آزمون کوتاه با `kubectl port-forward` به همان پاد انجام می‌شود. تابع زیر در صورت نبود پاد یا درگاه فقط پیام خطا چاپ می‌کند و با `exit` پوسته ورود را نمی‌بندد. پاسخ JSON مدل مستقیماً روی ترمینال چاپ می‌شود؛ این درخواست کوتاه فقط آزمون پاسخ است و اندازه‌گیری گذردهی نیست.

```bash
show_model() (
  # A subshell keeps variables and cleanup local to this call.
  PF_PID=
  WORK=$(mktemp -d) || return 1
  trap 'if [ -n "$PF_PID" ]; then kill "$PF_PID" 2>/dev/null || true; wait "$PF_PID" 2>/dev/null || true; fi; rm -rf "$WORK"' EXIT
  ARM="$1"
  ROLE="$2"
  API_KIND="$3"
  LOCAL_PORT=18000
  case "$ROLE:$API_KIND" in
    llm:chat|tp:chat|pp:chat) EXPECTED_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ ;;
    embed:embed) EXPECTED_MODEL=BAAI/bge-m3 ;;
    *) echo 'Use llm/tp/pp chat or embed embed'; return 1 ;;
  esac
  # Wait briefly for the controller to create the workload pod.
  POD=
  for _ in $(seq 1 30); do
    POD=$(kubectl -n llm-demo get pods \
    -l "app.kubernetes.io/name=pool-$ARM-$ROLE,kserve.io/component=workload" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [ -n "$POD" ] && break
    sleep 2
  done

  if [ -z "$POD" ]; then
    echo "No $ROLE pod found for $ARM"
    return 1
  fi

  if ! kubectl -n llm-demo wait --for=condition=Ready "pod/$POD" --timeout=900s; then
    echo "Pod $POD did not become Ready."
    return 1
  fi

  BASE=
  PF_PID=
  for REMOTE_PORT in 8000 8080 80; do
    kubectl -n llm-demo port-forward "pod/$POD" "$LOCAL_PORT:$REMOTE_PORT" \
      >"$WORK/port-forward.log" 2>&1 &
    PF_PID=$!

    for _ in 1 2 3 4 5; do
      sleep 1
      kill -0 "$PF_PID" 2>/dev/null || break
      if curl -fsS --max-time 3 "http://127.0.0.1:$LOCAL_PORT/v1/models" \
        >"$WORK/model.json" 2>/dev/null; then
        BASE="http://127.0.0.1:$LOCAL_PORT"
        break
      fi
    done

    if [ -n "$BASE" ]; then
      break
    fi

    kill "$PF_PID" 2>/dev/null || true
    wait "$PF_PID" 2>/dev/null || true
    PF_PID=
  done

  if [ -z "$BASE" ]; then
    echo "No model endpoint for $ARM/$ROLE"
    cat "$WORK/port-forward.log" 2>/dev/null || true
    return 1
  fi

  MODEL=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["data"][0]["id"])' "$WORK/model.json" \
    2>/dev/null)
  echo "ARM=$ARM ROLE=$ROLE POD=$POD MODEL=$MODEL"

  if [ "$MODEL" != "$EXPECTED_MODEL" ]; then
    echo "Wrong model for $ROLE: expected $EXPECTED_MODEL, got $MODEL"
    return 1
  elif [ "$API_KIND" = "chat" ]; then
    HTTP=$(curl -sS --max-time 30 -o "$WORK/response.json" -w '%{http_code}' \
      "$BASE/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain GPU sharing briefly.\"}],\"max_tokens\":8}") || return 1
  else
    HTTP=$(curl -sS --max-time 30 -o "$WORK/response.json" -w '%{http_code}' \
      "$BASE/v1/embeddings" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"input\":[\"GPU sharing demonstration\"]}") || return 1
  fi
  cat "$WORK/response.json"
  printf '\nHTTP %s\n' "$HTTP"
  [ "$HTTP" = 200 ] || return 1
  python3 - "$WORK/response.json" "$API_KIND" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert not d.get('error'), d
if sys.argv[2] == 'chat':
    assert d.get('choices') and d['choices'][0].get('message', {}).get('content'), d
else:
    assert d.get('data') and d['data'][0].get('embedding'), d
print('PASS: expected model returned a valid response')
PY
)
```

در حالت اختصاصی، خروجی هر دو خدمت با فرمان‌های زیر روی همان ترمینال چاپ می‌شود. در دو حالت بعدی همین تابع با نام حالت متناظر اجرا می‌شود.

```bash
show_model dedicated llm chat
show_model dedicated embed embed
```

فرمان‌های زیر تخصیص دستگاه و فرایندهای مقیم را نشان می‌دهند. در اشتراک تک‌دستگاهی، دو پاد باید شناسه مشترک داشته باشند؛ در تفکیک، هر نقش باید روی هر دو دستگاه مشاهده شود. شناسه تخصیص‌یافته با فرایندهای واقعی تطبیق داده می‌شود.

```bash
kubectl -n llm-demo get pods -o \
  'custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,GPU:.metadata.annotations.hami\.io/vgpu-devices-allocated'
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
nvidia-smi pmon -c 1
```

## ۶. تغییر از اختصاصی به اشتراک HAMI

دو خدمت اختصاصی حذف می‌شوند. پس از حذف پادهای آن‌ها و خالی‌شدن دستگاه‌ها، گروه به hami تغییر می‌کند و فایل‌های اشتراکی اعمال می‌شوند. MPS خاموش می‌ماند. سپس پاسخ هر دو مدل در همان حالت چاپ می‌شود. سهم‌بندی HAMI به معنی استفاده از MPS نیست و حضور دو پاد نیز به‌تنهایی اشتراک یک دستگاه را ثابت نمی‌کند [۵].

```bash
kubectl delete -f box1/dedicated/10-llm.yaml -f box1/dedicated/11-embed.yaml --ignore-not-found=true
kubectl -n llm-demo wait --for=delete pod \
  -l 'app.kubernetes.io/name in (pool-dedicated-llm,pool-dedicated-embed)' --timeout=180s
kubectl -n llm-demo get pods -o wide
nvidia-smi
```

پس از تأیید خروج مشتریان، مرحله بعد اجرا می‌شود.

```bash
kubectl label node mr-gholami gpu-share.io/pool=hami \
  gpu-share.io/mps-enabled=false --overwrite
kubectl apply --dry-run=server -f box1/hami/10-llm.yaml -f box1/hami/11-embed.yaml
kubectl apply -f box1/hami/10-llm.yaml -f box1/hami/11-embed.yaml
kubectl -n llm-demo get llminferenceservice,pods -o wide

show_model hami llm chat
show_model hami embed embed
```

## ۷. فعال‌سازی و توقف MPS

پس از نمایش HAMI، دو خدمت آن حذف و خروج مشتریان تأیید می‌شود. سپس گروه hami-mps و برچسب فعال‌سازی انتخاب می‌شوند. سرپرست، خدمت کنترل چندفرایندی را راه‌اندازی می‌کند. فایل‌های مدل، مسیر مشترک ارتباط و تنظیم سقف رشته فعال دارند. سقف رشته فعال با سهم سخت پردازنده فیزیکی برابر نیست؛ تغییر آن برای مشتری جدید اعمال می‌شود و جای تأیید اتصال واقعی را نمی‌گیرد [۴].

```bash
kubectl delete -f box1/hami/10-llm.yaml -f box1/hami/11-embed.yaml --ignore-not-found=true
kubectl -n llm-demo get pods -o wide
kubectl -n llm-demo wait --for=delete pod \
  -l 'app.kubernetes.io/name in (pool-hami-llm,pool-hami-embed)' --timeout=180s
nvidia-smi
```

پس از خروج مشتریان، سرپرست و سپس مدل‌ها فعال می‌شوند و پاسخ هر دو خدمت روی ترمینال چاپ می‌شود.

```bash
kubectl apply -f infrastructure/mps-pool.yaml
kubectl label node mr-gholami gpu-share.io/pool=hami-mps \
  gpu-share.io/mps-enabled=true --overwrite
kubectl -n llm-demo rollout status ds/qwen-benchmark-mps-daemon --timeout=180s
kubectl -n llm-demo wait --for=condition=Ready pod \
  -l app=qwen-benchmark-mps-daemon --timeout=180s
kubectl apply --dry-run=server -f box1/hami-mps/10-llm.yaml -f box1/hami-mps/11-embed.yaml
kubectl apply -f box1/hami-mps/10-llm.yaml -f box1/hami-mps/11-embed.yaml
kubectl -n llm-demo get llminferenceservice,pods -o wide

show_model hami-mps llm chat
show_model hami-mps embed embed

echo get_server_list | env CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
  nvidia-cuda-mps-control
nvidia-smi pmon -c 1
```

پس از پاسخ، فهرست کارساز MPS نباید خالی باشد. شناسه‌ها از خروجی `get_server_list` به‌صورت خودکار خوانده می‌شوند و برای هر کارساز، فهرست مشتریان چاپ می‌شود. در تفکیک مشترک، چهار کارگر مدل انتظار می‌رود. نام فرایند یا وجود سوکت بدون مشتری، اثبات مسیر MPS نیست. نسخه آزموده‌شده از شناسه کاربری یکسان استفاده می‌کند؛ اشتراک کاربران با شناسه متفاوت در این پیکربندی تضمین نشده است.

```bash
MPS_SERVERS=$(echo get_server_list \
  | env CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps nvidia-cuda-mps-control \
  | awk '$1 ~ /^[0-9]+$/ {print $1}')

if [ -z "$MPS_SERVERS" ]; then
  echo 'No MPS server found'
else
  for SERVER_PID in $MPS_SERVERS; do
    echo "MPS server: $SERVER_PID"
    echo "get_client_list $SERVER_PID" \
      | env CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps nvidia-cuda-mps-control
  done
fi
```

برای توقف کامل نمایش، هر کدام از سه جفت خدمت که وجود داشته باشد حذف می‌شود؛ سپس MPS خاموش می‌شود. این بلوک نیز در صورت نبود یک منبع پوسته را نمی‌بندد.

```bash
kubectl delete -f box1/hami-mps/10-llm.yaml -f box1/hami-mps/11-embed.yaml --ignore-not-found=true
kubectl delete -f box1/hami/10-llm.yaml -f box1/hami/11-embed.yaml --ignore-not-found=true
kubectl delete -f box1/dedicated/10-llm.yaml -f box1/dedicated/11-embed.yaml --ignore-not-found=true
kubectl -n llm-demo wait --for=delete pod \
  -l 'app.kubernetes.io/name in (pool-dedicated-llm,pool-dedicated-embed,pool-hami-llm,pool-hami-embed,pool-hami-mps-llm,pool-hami-mps-embed)' --timeout=180s
kubectl label node mr-gholami gpu-share.io/pool=dedicated \
  gpu-share.io/mps-enabled=false --overwrite
kubectl apply -f infrastructure/mps-pool.yaml
kubectl -n llm-demo get pods -o wide
kubectl -n llm-demo get pods -l app=qwen-benchmark-mps-daemon -o wide
pgrep -af nvidia-cuda-mps || true
nvidia-smi
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

فایل monitoring/servicemonitors.yaml هدف گردآوری و monitoring/dashboard.yaml داشبورد را تعریف می‌کند؛ این فایل‌ها سامانه پایش را نصب نمی‌کنند. این بخش فقط مسیر بازیابی است: اگر بررسی ابتدای راهنما نشان داد پایش بازگشته و آماده است، نصب و ارتقای این بخش اجرا نشود. در وضعیت قبلی Box1 انتشار پرومتئوس حذف شده بود؛ رابط HAMi نام `prometheus-kube-prometheus-prometheus.default.svc.cluster.local` را استفاده می‌کرد. نسخه ۶۲٫۶٫۰ در راهنمای بازیابی قبلی انتخاب شده است؛ این انتخاب را اندازه‌گیری عملکرد نسخه جدید معرفی نمی‌کنیم. خطای مالکیت Helm باید بررسی شود؛ اشیای انتشار دیگر خودکار تصاحب نمی‌شوند.

```bash
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts --force-update
helm repo update prometheus-community

PROM_CHART_VERSION=62.6.0

if helm -n default status prometheus >/dev/null 2>&1; then
  echo 'Prometheus Helm release already exists.'
else
  helm install prometheus prometheus-community/kube-prometheus-stack \
    --version "$PROM_CHART_VERSION" \
    -n default --create-namespace \
    --set alertmanager.enabled=false \
    --set grafana.enabled=true \
    --set grafana.sidecar.dashboards.searchNamespace=ALL \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
    --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false
fi

kubectl -n default patch prometheus prometheus-kube-prometheus-prometheus \
  --type=merge \
  -p '{"spec":{"replicas":1,"serviceMonitorSelector":{},"serviceMonitorNamespaceSelector":{},"podMonitorSelector":{},"podMonitorNamespaceSelector":{}}}'


kubectl -n default scale deployment/prometheus-kube-prometheus-operator --replicas=1
kubectl -n default scale deployment/prometheus-grafana --replicas=1

kubectl -n default get pods -o wide
kubectl -n default get svc prometheus-kube-prometheus-prometheus prometheus-operated prometheus-grafana
```

سپس هدف‌های سنجه و داشبورد پروژه اعمال می‌شوند. انتخابگر پرومتئوس در بلوک قبل برای ServiceMonitor و PodMonitor بدون محدودیت برچسب تنظیم شده است تا هدف‌های `llm-demo` و فضای نام `kube-system` نیز دیده شوند. همان هدف موجود به‌روزرسانی می‌شود تا گردآوری تکراری رخ ندهد. نام گروه و نقش در برچسب سنجه حفظ می‌شود.

```bash
kubectl apply --dry-run=server -f monitoring/servicemonitors.yaml \
  -f monitoring/dashboard.yaml
kubectl apply -f monitoring/servicemonitors.yaml -f monitoring/dashboard.yaml
kubectl apply -f monitoring/pool-workloads.yaml
kubectl get servicemonitor,podmonitor -A
```

رابط HAMI نسخه ۱٫۲٫۰ در خوشه فعلی نصب است ولی Deployment آن متوقف شده بود. این بلوک همان انتشار را به یک replica برمی‌گرداند و آدرس پرومتئوس را صریحاً به خدمت بازگردانده‌شده در default متصل می‌کند. اگر صادرکننده DCGM از قبل توسط GPU Operator یا نصب دیگری وجود داشته باشد، صادرکننده تکراری رابط HAMI غیرفعال می‌شود.

```bash
helm repo add hami-webui https://project-hami.github.io/HAMi-WebUI --force-update
helm repo update hami-webui

DCGM_ENABLED=true
if kubectl get clusterpolicy cluster-policy >/dev/null 2>&1; then
  kubectl patch clusterpolicy cluster-policy --type=merge \
    -p '{"spec":{"dcgmExporter":{"enabled":true}}}'
  DCGM_ENABLED=false
fi

if helm -n kube-system status my-hami-webui >/dev/null 2>&1; then
  helm upgrade my-hami-webui hami-webui/hami-webui \
    --version 1.2.0 -n kube-system --reuse-values \
    --set replicaCount=1 \
    --set kube-prometheus-stack.enabled=false \
    --set dcgm-exporter.enabled="$DCGM_ENABLED" \
    --set externalPrometheus.enabled=true \
    --set-string externalPrometheus.address=http://prometheus-kube-prometheus-prometheus.default.svc.cluster.local:9090
else
  helm install my-hami-webui hami-webui/hami-webui \
    --version 1.2.0 -n kube-system \
    --set replicaCount=1 \
    --set kube-prometheus-stack.enabled=false \
    --set dcgm-exporter.enabled="$DCGM_ENABLED" \
    --set externalPrometheus.enabled=true \
    --set-string externalPrometheus.address=http://prometheus-kube-prometheus-prometheus.default.svc.cluster.local:9090
fi

kubectl -n kube-system rollout status deployment/my-hami-webui --timeout=300s
kubectl -n kube-system get deployment my-hami-webui
kubectl -n kube-system get svc my-hami-webui -o wide
kubectl -n default get endpoints prometheus-kube-prometheus-prometheus
kubectl -n kube-system logs deployment/my-hami-webui --all-containers --tail=30 2>/dev/null || true
```

در نمایش، نام حالت، بازه زمانی جاری، جای‌گذاری واقعی و پاسخ مدل‌ها نشان داده می‌شود. نمودارهای پژوهش از نتایج ذخیره‌شده خوانده می‌شوند؛ چند درخواست کوتاه آزمون ظرفیت نیستند. شواهد اجرای مدل از آزمایش‌های ثبت‌شده و پیکربندی‌های اصلی همراه بسته به دست آمده است. بررسی‌های عملیاتی شامل پاسخ هر دو مدل، نبود بازراه‌اندازی، تطبیق شناسه دستگاه‌ها، وضعیت درست MPS و پایش فعال است. نام گره، مخزن تصویر، مسیر وزن و ظرفیت تخصیص‌پذیر، مقادیر وابسته به محیط مقصد هستند [۱۳]، [۱۴].

## 10. نمایش کوتاه جای‌گذاری، سنجه و داشبورد

روی خود گره گرافیکی، فرمان زیر برای ۲۰ ثانیه وضعیت دستگاه‌ها را نشان می‌دهد. هم‌زمان، یک درخواست بخش ۵ ارسال کنید؛ سپس شناسه تخصیص پاد، شناسه دستگاه و فرایند مدل را کنار هم نمایش دهید. پایان زمان فرمان timeout طبیعی است. بالا رفتن میزان استفاده از دستگاه به‌تنهایی اثبات هم‌مکانی دو مدل نیست.

```bash
timeout 20s nvidia-smi --loop=1
kubectl -n llm-demo get pods -o \
  'custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,GPU:.metadata.annotations.hami\.io/vgpu-devices-allocated'
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

زنجیره پایش چنین است: نقطه /metrics زمان اجرا و صادرکننده‌های HAMI و انویدیا، هدف‌های PodMonitor و ServiceMonitor، پرومتئوس، و سپس گرافانا یا رابط HAMI. برای نشان‌دادن سهم تنظیم‌شده، بخش resources فایل مدل و حاشیه‌نویسی تخصیص را باز کنید؛ برای مصرف و عملکرد واقعی، سنجه‌های گردآوری‌شده را نشان دهید. بخش ۹ سامانه پایش و فایل‌های هدف را فعال می‌کند.

```bash
curl -fsS --max-time 5 http://127.0.0.1:31993/metrics \
  | grep -E '^hami_gpu_(core|memory)' | head -n 12
kubectl get servicemonitor,podmonitor -A
```

برای مرورگر، هر انتقال درگاه را در یک پایانه جدا اجرا کنید. این فرمان‌ها `exit` ندارند و با Ctrl+C فقط همان انتقال درگاه متوقف می‌شود. پس از بخش ۹، نام خدمت‌ها ثابت و قابل بررسی است.

```bash
kubectl -n default port-forward \
  svc/prometheus-kube-prometheus-prometheus 9090:9090
```

در مرورگر، `http://127.0.0.1:9090/targets` را باز کنید؛ هدف زمان اجرای مدل، HAMI و صادرکننده GPU باید فعال باشند.

```bash
kubectl -n default get secret prometheus-grafana \
  -o jsonpath='{.data.admin-password}' | base64 -d; echo
kubectl -n default port-forward svc/prometheus-grafana 3000:80
```

نام کاربری پیش‌فرض گرافانا `admin` است؛ فرمان قبل گذرواژه همان انتشار را چاپ می‌کند. در مرورگر، `http://127.0.0.1:3000/d/r2-gpu-sharing` را باز کنید. سپس رابط HAMI را در پایانه سوم باز کنید.

فرمان نمایش گذرواژه پیش از شروع ضبط اجرا شود. `127.0.0.1` مربوط به رایانه‌ای است که port-forward روی آن اجرا شده است. اگر پایانه روی Box1 و مرورگر روی رایانه شخصی است، در بخش Ports محیط توسعه همان درگاه‌های ۹۰۹۰، ۳۰۰۰ و ۳۰۰۱ را به رایانه مرورگر منتقل کنید؛ سرویس‌ها را برای این ویدئو عمومی نکنید.

```bash
kubectl -n kube-system port-forward svc/my-hami-webui 3001:3000
```

برای توقف صریح انتقال درگاه‌های ویدئو، فرمان‌های زیر فقط فرایندهای `port-forward` همین سه داشبورد را متوقف می‌کنند؛ نبودن هر فرایند خطا محسوب نمی‌شود و پوسته ورود بسته نمی‌شود.

```bash
pkill -f 'kubectl -n default port-forward.*prometheus-kube-prometheus-prometheus.*9090:9090' 2>/dev/null || true
pkill -f 'kubectl -n default port-forward.*prometheus-grafana.*3000:80' 2>/dev/null || true
pkill -f 'kubectl -n kube-system port-forward.*my-hami-webui.*3001:3000' 2>/dev/null || true
```

صفحه `http://127.0.0.1:3001` برای نمایش گره، دستگاه و سهم پادها باز می‌شود. خطای `no such host` برای `prometheus-kube-prometheus-prometheus.default.svc.cluster.local` نشان می‌دهد خدمت پرومتئوس بخش ۹ هنوز ساخته یا آماده نشده است؛ در این حالت ابتدا وضعیت Service و Endpoint همان نام بررسی می‌شود، نه اینکه انتقال درگاه باعث نصب سرویس فرض شود.

```bash
kubectl -n default get svc prometheus-kube-prometheus-prometheus
kubectl -n default get endpoints prometheus-kube-prometheus-prometheus
kubectl -n kube-system get pods -l app.kubernetes.io/instance=my-hami-webui -o wide
```

سنجه‌های زیر در بخش جست‌وجوی پرومتئوس یا کاوش گرافانا قابل نمایش‌اند. دو درخواست کوتاه برای نتیجه‌گیری درباره صدک‌ها کافی نیستند؛ نمودارهای تأخیر پژوهش از اجرای ثبت‌شده خوانده می‌شوند. نبود سری سنجه با مقدار صفر یکسان نیست؛ ابتدا فعال‌بودن هدف گردآوری و وجود نام سنجه بررسی شود.

```promql
hami_gpu_memory_allocated_bytes
hami_gpu_core_allocated_ratio
sum by (r2_role) (vllm:num_requests_running{namespace="llm-demo"})
sum by (r2_role) (rate(vllm:generation_tokens_total{namespace="llm-demo"}[1m]))
```

بهترین ضبط کوتاه، یک پاسخ گفت‌وگو، یک پاسخ تعبیه‌سازی و حدود ۲۰ تا ۳۰ ثانیه نمایش جای‌گذاری و داشبورد است؛ دانلود وزن و آماده‌سازی اولیه مدل جزو زمان این آزمون کوتاه نیست.

## ۱۱. معرفی کوتاه GPU Operator و نصب موجود

روی Box1 نصب فعلی را نشان دهید. GPU Operator مدیریت اجزای زیرساخت را انجام می‌دهد؛ تخصیص GPU در این آزمایش با device plugin متعلق به HAMi انجام شده است. انتشار اپراتور موجود دوباره نصب نشود. نصب MPS نیز در این مسیر همان اعمال DaemonSet بخش ۷ است و نصب سامانه سروینگ جدیدی نیست.

```bash
helm -n gpu-operator list
kubectl get clusterpolicy cluster-policy
kubectl -n gpu-operator get pods -o wide
kubectl get ds -A
helm -n kube-system get values hami -a
```

برای معرفی روش نصب در مقصد تازه، صفحه رسمی GPU Operator و بخش ۲ نصب HAMi نشان داده شود. نصب‌کننده مقصد باید مالک فعلی driver، toolkit و device plugin را مشخص کند؛ تنظیم‌های نصب تازه را به نصب فعال Box1 تحمیل نکنید. مرجع: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.3/getting-started.html

## ۱۲. سازگاری پایش با نسخه آزموده‌شده و بررسی پیش از ضبط

این بخش پس از وجود پرومتئوس و پیش از شروع مدل‌ها اجرا می‌شود. اگر بخش ۹ اجرا شده است، این اصلاح‌ها بعد از آن باشند؛ اعمال دوباره فایل اصلی پایش، اصلاح‌های زیر را جایگزین می‌کند. نام سنجه‌ها از `prom.json` آزمایش ۱۵ و جمع‌آورنده round2 گرفته شده است. نام‌های قدیمی `gpu_cache_usage_perc` و `time_per_output_token_seconds_bucket` در داشبورد با `kv_cache_usage_perc` و `inter_token_latency_seconds_bucket` جایگزین می‌شوند. پنل Prefix cache که داده ثبت‌شده ندارد از نمای ضبط حذف می‌شود. YAMLهای اصلی آزمایش تغییر نمی‌کنند.

```bash
kubectl apply -f monitoring/servicemonitors.yaml -f monitoring/dashboard.yaml
kubectl apply -f monitoring/pool-workloads.yaml

# Select the Service's named port, not the node-facing port as a pod targetPort.
kubectl -n kube-system get svc hami-scheduler -o yaml
kubectl -n kube-system patch servicemonitor r2-hami-scheduler --type=json \
  -p='[{"op":"replace","path":"/spec/endpoints/0","value":{"port":"monitor","path":"/metrics","interval":"5s","scrapeTimeout":"4s","honorLabels":true}}]'
```

پورت `monitor` در chart نسخه ۲٫۹٫۰ به پورت سرویس ۳۱۹۹۳ و سپس پورت داخلی ۹۳۹۵ متصل است؛ مقدار NodePort نباید به‌عنوان پورت داخلی پاد استفاده شود. فایل ServiceMonitor اصلی در فضای نام kube-system قرار دارد. برای DCGM فقط یک صادرکننده فعال انتخاب شود. بلوک بعد روی Box1 شناخته‌شده، صادرکننده GPU Operator را در اولویت قرار می‌دهد؛ اگر این DaemonSet وجود نداشته باشد، انتخابگر ثبت‌شده صادرکننده WebUI حفظ می‌شود.

```bash
if kubectl -n gpu-operator get ds nvidia-dcgm-exporter >/dev/null 2>&1; then
  kubectl -n gpu-operator rollout status ds/nvidia-dcgm-exporter --timeout=180s
  kubectl -n gpu-operator get svc nvidia-dcgm-exporter --show-labels
  kubectl -n kube-system patch servicemonitor r2-dcgm --type=json \
    -p='[{"op":"replace","path":"/spec/namespaceSelector/matchNames","value":["gpu-operator"]},{"op":"replace","path":"/spec/selector","value":{"matchLabels":{"app":"nvidia-dcgm-exporter"}}}]'
else
  kubectl -n kube-system get ds,pods,svc -l app.kubernetes.io/instance=my-hami-webui
fi

python3 - <<'PY'
import json, subprocess
raw = subprocess.check_output([
    'kubectl', '-n', 'default', 'get', 'configmap',
    'r2-gpu-sharing-dashboard', '-o', 'json'], text=True)
cm = json.loads(raw)
dashboard = json.loads(cm['data']['r2-gpu-sharing.json'])
def repair(panels):
    result = []
    for panel in panels:
        if any('prefix_cache_hit_rate' in t.get('expr', '') for t in panel.get('targets', [])):
            continue
        for target in panel.get('targets', []):
            target['expr'] = target.get('expr', '').replace(
                'vllm:gpu_cache_usage_perc', 'vllm:kv_cache_usage_perc').replace(
                'vllm:time_per_output_token_seconds_bucket',
                'vllm:inter_token_latency_seconds_bucket')
        if 'panels' in panel:
            panel['panels'] = repair(panel['panels'])
        result.append(panel)
    return result
dashboard['panels'] = repair(dashboard['panels'])
patch = {'data': {'r2-gpu-sharing.json': json.dumps(dashboard, ensure_ascii=False)}}
subprocess.run(['kubectl', '-n', 'default', 'patch', 'configmap',
    'r2-gpu-sharing-dashboard', '--type=merge', '-p', json.dumps(patch)], check=True)
PY

kubectl -n default rollout status deployment/prometheus-grafana --timeout=180s
kubectl -n kube-system rollout status deployment/my-hami-webui --timeout=180s
kubectl -n default get endpoints prometheus-kube-prometheus-prometheus prometheus-grafana
kubectl -n kube-system get endpoints my-hami-webui hami-scheduler
kubectl get servicemonitor,podmonitor -A
```

پس از port-forward بخش ۱۰، در پرومتئوس صفحه `/targets` را باز کنید. هدف‌های HAMi و DCGM باید UP باشند؛ هدف vLLM پس از آماده‌شدن مدل ایجاد می‌شود. پس از توقف یک حالت و شروع حالت بعد، بازه زمانی داشبورد را به همان مرحله محدود کنید. `r2_role` برای مدل‌های بسته شامل نام حالت است؛ برای نمونه `hami-llm` و `hami-mps-embed`. در Grafana منبع داده Prometheus و گره mr-gholami انتخاب شود. اگر پنل قدیمی ماند، صفحه را تازه‌سازی و وضعیت dashboard sidecar را بررسی کنید.

برای نمایش مستقیم نقطه تولید سنجه، این دو انتقال درگاه در دو پایانه جدا و هنگام آماده‌بودن مدل HAMi اجرا شوند. در حالت‌های دیگر نام `pool-hami-llm` با خدمت همان حالت جایگزین شود.

```bash
kubectl -n kube-system port-forward svc/hami-scheduler 19395:monitor
```

```bash
POD=$(kubectl -n llm-demo get pods \
  -l 'app.kubernetes.io/name=pool-hami-llm,kserve.io/component=workload' \
  -o jsonpath='{.items[0].metadata.name}')
kubectl -n llm-demo port-forward "pod/$POD" 18080:8000
```

در پایانه سوم، روی همان میزبان port-forward:

```bash
curl -fsS --max-time 5 http://127.0.0.1:19395/metrics \
  | grep -E '^hami_gpu_(core|memory)' | head -n 12
curl -fsS --max-time 5 http://127.0.0.1:18080/metrics \
  | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|generation_tokens_total)' | head -n 12
```

هر عبارت زیر جداگانه در Explore یا Prometheus اجرا شود. این‌ها نام‌ها و queryهای ثبت‌شده آزمایش هستند؛ ضرب در ۱۰۰ فقط تبدیل KV cache از نسبت به درصد است. سنجه‌های HAMi تخصیص هستند، نه مصرف واقعی. `rate` و histogram به چند scrape و درخواست کافی نیاز دارند؛ نمودار صفر یا خالی پس از یک درخواست کوتاه، نتیجه بنچمارک نیست.

```promql
hami_gpu_memory_allocated_bytes
hami_gpu_core_allocated_ratio
hami_gpu_core_limit_ratio
DCGM_FI_DEV_GPU_UTIL
DCGM_FI_DEV_FB_USED
DCGM_FI_DEV_MEM_COPY_UTIL
DCGM_FI_DEV_SM_CLOCK
DCGM_FI_DEV_POWER_USAGE
DCGM_FI_DEV_GPU_TEMP
sum by (r2_role) (vllm:num_requests_running{namespace="llm-demo"})
sum by (r2_role) (vllm:num_requests_waiting{namespace="llm-demo"})
vllm:kv_cache_usage_perc{namespace="llm-demo"} * 100
sum by (r2_role) (rate(vllm:num_preemptions_total{namespace="llm-demo"}[1m]))
sum by (r2_role) (rate(vllm:generation_tokens_total{namespace="llm-demo"}[1m]))
histogram_quantile(0.95, sum by (le, r2_role) (rate(vllm:time_to_first_token_seconds_bucket{namespace="llm-demo"}[1m])))
histogram_quantile(0.99, sum by (le, r2_role) (rate(vllm:e2e_request_latency_seconds_bucket{namespace="llm-demo"}[1m])))
```

## ۱۳. اگر هنگام ضبط چیزی کار نکرد

| نشانه | بررسی محدود و اقدام |
|---|---|
| پاد Pending | برچسب همان حالت روی mr-gholami، درخواست RAM و رویدادهای پاد بررسی شود؛ برچسب نام گره فرضی استفاده نشود |
| مدل embedding پاسخ ۴۰۰ داد | نام BAAI/bge-m3 و آرگومان‌های pooling/convert در 11-embed بررسی شود؛ درخواست embedding به مدل Qwen ارسال نشود |
| No route to host | آزمون بخش ۵ از port-forward استفاده می‌کند؛ Pod IP قدیمی دوباره استفاده نشود |
| MPS روشن ولی client خالی است | هر دو مدل باید آماده و پاسخ‌گو باشند؛ get_server_list و get_client_list بخش ۷ بررسی شود |
| Grafana باز است ولی نمودار ندارد | منبع داده، بازه زمان، UP بودن targets، نام سنجه و r2_role بررسی شود؛ رابط باز به معنی گردآوری داده نیست |
| HAMi Dashboard باز است ولی داده ندارد | اتصال backend به Prometheus، endpoint پرومتئوس و وجود سری تخصیص بررسی شود |
| فقط GPU chart خالی است | صادرکننده انتخاب‌شده، ServiceMonitor r2-dcgm و هدف DCGM بررسی شود؛ بخش ۱۲ |
| خطای مالکیت Helm در بازیابی | نصب متوقف شود؛ اشیای متعلق به انتشار دیگر خودکار تصاحب نشوند |

```bash
kubectl -n llm-demo get pods -o wide
kubectl -n llm-demo get events --sort-by=.lastTimestamp
kubectl describe node mr-gholami
kubectl -n default logs deployment/prometheus-grafana -c grafana-sc-dashboard --tail=30
kubectl -n kube-system logs deployment/my-hami-webui --all-containers --tail=30
```

منشأ فرمان‌های پاسخ، انتخاب پاد و API: harness و replay همان round2؛ منشأ YAMLهای مدل: نسخه‌های ثبت‌شده T1 و نگاشت‌های pool-provenance؛ منشأ MPS: سرپرست اصلی و بررسی مشتریان آزمایش؛ منشأ queryها: prom.json و dashboard آزمایش‌ها. انتظار حذف پاد، کنترل HTTP و اصلاح اتصال پایش، اصلاح‌های عملیاتی همین راهنما هستند؛ آن‌ها اجرای بنچمارک جدید محسوب نمی‌شوند. اعتبارسنجی محلی شامل نحو و منطق است و جای مشاهده وضعیت زنده Box1 را نمی‌گیرد.
