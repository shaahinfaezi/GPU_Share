# شرح تک‌تک پیکربندی‌ها

مقادیر زیر مستقیماً از فایل اصلی خوانده شده‌اند. سه حالت یک خانواده، جایگزین یکدیگرند؛ اعمال هم‌زمان همه پوشه‌ها به‌دلیل نام‌های مشترک صحیح نیست. برای حالت اختصاصی فقط یک خدمت مقیم باشد. پیکربندی اشتراکی هر خانواده شامل دو خدمت است.

دستورها و نام کلیدها برای جلوگیری از تغییر معنای اجرایی، به شکل اصلی حفظ شده‌اند. نام مدل و تصویر نیز شناسه فنی هستند و ترجمه نمی‌شوند.

## configurations/llm-embedding/dedicated/r2-dedicated-embed.yaml

این فایل خدمت «r2-embed» را برای مدل «BAAI/bge-m3» تعریف می‌کند. منشأ: `T1-20260902-144618/dedicated__embed/rendered/r2-dedicated-embed.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: BAAI/bge-m3
  uri: pvc://r2-models-pvc/models/embed
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.90
- --max-model-len=8192
- --max-num-seqs=64
- --runner=pooling
- --convert=embed
- --dtype=half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/llm-embedding/dedicated/r2-dedicated-llm.yaml

این فایل خدمت «r2-llm» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `T1-20260902-144618/dedicated__llm/rendered/r2-dedicated-llm.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.92
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/llm-embedding/hami-mps/r2-hami-mps-embed.yaml

این فایل خدمت «r2-embed» را برای مدل «BAAI/bge-m3» تعریف می‌کند. منشأ: `T1-20260902-144618/hami-mps__llm+embed/rendered/r2-hami-mps-embed.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: BAAI/bge-m3
  uri: pvc://r2-models-pvc/models/embed
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.90
- --max-model-len=8192
- --max-num-seqs=64
- --runner=pooling
- --convert=embed
- --dtype=half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: CUDA_MPS_PIPE_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_LOG_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: '100'
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/llm-embedding/hami-mps/r2-hami-mps-llm.yaml

این فایل خدمت «r2-llm» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `T1-20260902-144618/hami-mps__llm+embed/rendered/r2-hami-mps-llm.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.92
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: CUDA_MPS_PIPE_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_LOG_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: '100'
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/llm-embedding/hami/r2-hami-embed.yaml

این فایل خدمت «r2-embed» را برای مدل «BAAI/bge-m3» تعریف می‌کند. منشأ: `T1-20260902-144618/hami__llm+embed/rendered/r2-hami-embed.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: BAAI/bge-m3
  uri: pvc://r2-models-pvc/models/embed
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.90
- --max-model-len=8192
- --max-num-seqs=64
- --runner=pooling
- --convert=embed
- --dtype=half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/llm-embedding/hami/r2-hami-llm.yaml

این فایل خدمت «r2-llm» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `T1-20260902-144618/hami__llm+embed/rendered/r2-hami-llm.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.92
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/two-llms/dedicated/r2-dedicated-llm.yaml

این فایل خدمت «r2-llm» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `T2-20260902-152143/dedicated__llm/rendered/r2-dedicated-llm.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.92
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/two-llms/dedicated/r2-dedicated-llm2.yaml

این فایل خدمت «r2-llm2» را برای مدل «Qwen/Qwen2.5-1.5B-Instruct-AWQ» تعریف می‌کند. منشأ: `T2-20260902-152143/dedicated__llm2/rendered/r2-dedicated-llm2.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm2
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.79
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3840'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3840'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/two-llms/hami-mps/r2-hami-mps-llm.yaml

این فایل خدمت «r2-llm» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `T2-20260902-152143/hami-mps__llm+llm2/rendered/r2-hami-mps-llm.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.92
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: CUDA_MPS_PIPE_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_LOG_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: '100'
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/two-llms/hami-mps/r2-hami-mps-llm2.yaml

این فایل خدمت «r2-llm2» را برای مدل «Qwen/Qwen2.5-1.5B-Instruct-AWQ» تعریف می‌کند. منشأ: `T2-20260902-152143/hami-mps__llm+llm2/rendered/r2-hami-mps-llm2.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm2
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.79
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3840'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3840'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: CUDA_MPS_PIPE_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_LOG_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: '100'
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/two-llms/hami/r2-hami-llm.yaml

این فایل خدمت «r2-llm» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `T2-20260902-152143/hami__llm+llm2/rendered/r2-hami-llm.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.92
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/two-llms/hami/r2-hami-llm2.yaml

این فایل خدمت «r2-llm2» را برای مدل «Qwen/Qwen2.5-1.5B-Instruct-AWQ» تعریف می‌کند. منشأ: `T2-20260902-152143/hami__llm+llm2/rendered/r2-hami-llm2.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm2
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.79
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3840'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3840'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/vision-reranker/dedicated/r2-dedicated-rerank.yaml

این فایل خدمت «r2-rerank» را برای مدل «BAAI/bge-reranker-v2-m3» تعریف می‌کند. منشأ: `T3-20260902-155701/dedicated__rerank/rendered/r2-dedicated-rerank.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: BAAI/bge-reranker-v2-m3
  uri: pvc://r2-models-pvc/models/rerank
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.88
- --max-model-len=8192
- --max-num-seqs=64
- --runner=pooling
- --convert=classify
- --dtype=half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '2816'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '2816'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/vision-reranker/dedicated/r2-dedicated-vl.yaml

این فایل خدمت «r2-vl» را برای مدل «Qwen/Qwen2-VL-2B-Instruct» تعریف می‌کند. منشأ: `T3-20260902-155701/dedicated__vl/rendered/r2-dedicated-vl.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2-VL-2B-Instruct
  uri: pvc://r2-models-pvc/models/vl
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.86
- --max-model-len=4096
- --max-num-seqs=32
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '9216'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '9216'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/vision-reranker/hami-mps/r2-hami-mps-rerank.yaml

این فایل خدمت «r2-rerank» را برای مدل «BAAI/bge-reranker-v2-m3» تعریف می‌کند. منشأ: `T3-20260902-155701/hami-mps__vl+rerank/rendered/r2-hami-mps-rerank.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: BAAI/bge-reranker-v2-m3
  uri: pvc://r2-models-pvc/models/rerank
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.88
- --max-model-len=8192
- --max-num-seqs=64
- --runner=pooling
- --convert=classify
- --dtype=half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '2816'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '2816'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: CUDA_MPS_PIPE_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_LOG_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: '100'
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/vision-reranker/hami-mps/r2-hami-mps-vl.yaml

این فایل خدمت «r2-vl» را برای مدل «Qwen/Qwen2-VL-2B-Instruct» تعریف می‌کند. منشأ: `T3-20260902-155701/hami-mps__vl+rerank/rendered/r2-hami-mps-vl.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2-VL-2B-Instruct
  uri: pvc://r2-models-pvc/models/vl
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.86
- --max-model-len=4096
- --max-num-seqs=32
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '9216'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '9216'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: CUDA_MPS_PIPE_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_LOG_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: '100'
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/vision-reranker/hami/r2-hami-rerank.yaml

این فایل خدمت «r2-rerank» را برای مدل «BAAI/bge-reranker-v2-m3» تعریف می‌کند. منشأ: `T3-20260902-155701/hami__vl+rerank/rendered/r2-hami-rerank.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: BAAI/bge-reranker-v2-m3
  uri: pvc://r2-models-pvc/models/rerank
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.88
- --max-model-len=8192
- --max-num-seqs=64
- --runner=pooling
- --convert=classify
- --dtype=half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '2816'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '2816'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/vision-reranker/hami/r2-hami-vl.yaml

این فایل خدمت «r2-vl» را برای مدل «Qwen/Qwen2-VL-2B-Instruct» تعریف می‌کند. منشأ: `T3-20260902-155701/hami__vl+rerank/rendered/r2-hami-vl.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2-VL-2B-Instruct
  uri: pvc://r2-models-pvc/models/vl
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.86
- --max-model-len=4096
- --max-num-seqs=32
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '9216'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '9216'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/symmetric-llms/dedicated/r2-dedicated-llm2a.yaml

این فایل خدمت «r2-llm2a» را برای مدل «Qwen/Qwen2.5-1.5B-Instruct-AWQ» تعریف می‌کند. منشأ: `T17-20260912-123151/dedicated__llm2a/rendered/r2-dedicated-llm2a.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm2
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.85
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/symmetric-llms/dedicated/r2-dedicated-llm2b.yaml

این فایل خدمت «r2-llm2b» را برای مدل «Qwen/Qwen2.5-1.5B-Instruct-AWQ» تعریف می‌کند. منشأ: `T17-20260912-123151/dedicated__llm2b/rendered/r2-dedicated-llm2b.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm2
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.85
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/symmetric-llms/hami-mps/r2-hami-mps-llm2a.yaml

این فایل خدمت «r2-llm2a» را برای مدل «Qwen/Qwen2.5-1.5B-Instruct-AWQ» تعریف می‌کند. منشأ: `T17-20260912-123151/hami-mps__llm2a+llm2b/rendered/r2-hami-mps-llm2a.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm2
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.85
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: CUDA_MPS_PIPE_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_LOG_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: '100'
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/symmetric-llms/hami-mps/r2-hami-mps-llm2b.yaml

این فایل خدمت «r2-llm2b» را برای مدل «Qwen/Qwen2.5-1.5B-Instruct-AWQ» تعریف می‌کند. منشأ: `T17-20260912-123151/hami-mps__llm2a+llm2b/rendered/r2-hami-mps-llm2b.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm2
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.85
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: CUDA_MPS_PIPE_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_LOG_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: '100'
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/symmetric-llms/hami/r2-hami-llm2a.yaml

این فایل خدمت «r2-llm2a» را برای مدل «Qwen/Qwen2.5-1.5B-Instruct-AWQ» تعریف می‌کند. منشأ: `T17-20260912-123151/hami__llm2a+llm2b/rendered/r2-hami-llm2a.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm2
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.85
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/symmetric-llms/hami/r2-hami-llm2b.yaml

این فایل خدمت «r2-llm2b» را برای مدل «Qwen/Qwen2.5-1.5B-Instruct-AWQ» تعریف می‌کند. منشأ: `T17-20260912-123151/hami__llm2a+llm2b/rendered/r2-hami-llm2b.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm2
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --gpu-memory-utilization=0.85
- --max-model-len=4096
- --max-num-seqs=64
- --quantization=awq_marlin
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/split-tp2-pp2/dedicated__tp/tp.yaml

این فایل خدمت «r2-tp» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `S-qwen7b-attempt-20260925-134248/dedicated__tp/rendered/tp.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --model
- /mnt/models
- --served-model-name
- Qwen/Qwen2.5-7B-Instruct-AWQ
- --host
- 0.0.0.0
- --port
- '8000'
- --tensor-parallel-size
- '2'
- --pipeline-parallel-size
- '1'
- --max-num-seqs
- '2'
- --max-model-len
- '1024'
- --enforce-eager
- --max-num-batched-tokens
- '256'
- --enable-chunked-prefill
- --distributed-executor-backend
- mp
- --gpu-memory-utilization
- '0.7'
- --kv-cache-memory-bytes
- '268435456'
- --no-enable-prefix-caching
- --safetensors-load-strategy
- lazy
- --load-format
- safetensors
- --quantization
- awq_marlin
- --dtype
- half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
env:
- name: HF_HOME
  value: /mnt/models
- name: VLLM_ADDITIONAL_ARGS
  value: --tensor-parallel-size 2 --pipeline-parallel-size 1 --max-num-seqs 2 --max-model-len
    1024 --enforce-eager --max-num-batched-tokens 256 --enable-chunked-prefill --distributed-executor-backend
    mp --gpu-memory-utilization 0.7 --kv-cache-memory-bytes 268435456 --no-enable-prefix-caching
    --safetensors-load-strategy lazy --load-format safetensors --quantization awq_marlin
    --dtype half
- name: MALLOC_ARENA_MAX
  value: '2'
- name: TOKENIZERS_PARALLELISM
  value: 'false'
- name: OMP_NUM_THREADS
  value: '1'
- name: CUDA_MODULE_LOADING
  value: LAZY
- name: VLLM_WORKER_MULTIPROC_METHOD
  value: fork
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/split-tp2-pp2/dedicated__pp/pp.yaml

این فایل خدمت «r2-pp» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `S-qwen7b-attempt-20260925-134248/dedicated__pp/rendered/pp.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --model
- /mnt/models
- --served-model-name
- Qwen/Qwen2.5-7B-Instruct-AWQ
- --host
- 0.0.0.0
- --port
- '8000'
- --tensor-parallel-size
- '1'
- --pipeline-parallel-size
- '2'
- --max-num-seqs
- '2'
- --max-model-len
- '1024'
- --enforce-eager
- --max-num-batched-tokens
- '256'
- --enable-chunked-prefill
- --distributed-executor-backend
- mp
- --gpu-memory-utilization
- '0.7'
- --kv-cache-memory-bytes
- '268435456'
- --no-enable-prefix-caching
- --safetensors-load-strategy
- lazy
- --load-format
- safetensors
- --quantization
- awq_marlin
- --dtype
- half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
env:
- name: HF_HOME
  value: /mnt/models
- name: VLLM_ADDITIONAL_ARGS
  value: --tensor-parallel-size 1 --pipeline-parallel-size 2 --max-num-seqs 2 --max-model-len
    1024 --enforce-eager --max-num-batched-tokens 256 --enable-chunked-prefill --distributed-executor-backend
    mp --gpu-memory-utilization 0.7 --kv-cache-memory-bytes 268435456 --no-enable-prefix-caching
    --safetensors-load-strategy lazy --load-format safetensors --quantization awq_marlin
    --dtype half
- name: MALLOC_ARENA_MAX
  value: '2'
- name: TOKENIZERS_PARALLELISM
  value: 'false'
- name: OMP_NUM_THREADS
  value: '1'
- name: CUDA_MODULE_LOADING
  value: LAZY
- name: VLLM_WORKER_MULTIPROC_METHOD
  value: fork
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/split-tp2-pp2/hami/pp.yaml

این فایل خدمت «r2-pp» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `S-qwen7b-attempt-20260925-134248/hami/rendered/pp.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --model
- /mnt/models
- --served-model-name
- Qwen/Qwen2.5-7B-Instruct-AWQ
- --host
- 0.0.0.0
- --port
- '8000'
- --tensor-parallel-size
- '1'
- --pipeline-parallel-size
- '2'
- --max-num-seqs
- '2'
- --max-model-len
- '1024'
- --enforce-eager
- --max-num-batched-tokens
- '256'
- --enable-chunked-prefill
- --distributed-executor-backend
- mp
- --gpu-memory-utilization
- '0.7'
- --kv-cache-memory-bytes
- '268435456'
- --no-enable-prefix-caching
- --safetensors-load-strategy
- lazy
- --load-format
- safetensors
- --quantization
- awq_marlin
- --dtype
- half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: VLLM_ADDITIONAL_ARGS
  value: --tensor-parallel-size 1 --pipeline-parallel-size 2 --max-num-seqs 2 --max-model-len
    1024 --enforce-eager --max-num-batched-tokens 256 --enable-chunked-prefill --distributed-executor-backend
    mp --gpu-memory-utilization 0.7 --kv-cache-memory-bytes 268435456 --no-enable-prefix-caching
    --safetensors-load-strategy lazy --load-format safetensors --quantization awq_marlin
    --dtype half
- name: MALLOC_ARENA_MAX
  value: '2'
- name: TOKENIZERS_PARALLELISM
  value: 'false'
- name: OMP_NUM_THREADS
  value: '1'
- name: CUDA_MODULE_LOADING
  value: LAZY
- name: VLLM_WORKER_MULTIPROC_METHOD
  value: fork
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/split-tp2-pp2/hami/tp.yaml

این فایل خدمت «r2-tp» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `S-qwen7b-attempt-20260925-134248/hami/rendered/tp.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --model
- /mnt/models
- --served-model-name
- Qwen/Qwen2.5-7B-Instruct-AWQ
- --host
- 0.0.0.0
- --port
- '8000'
- --tensor-parallel-size
- '2'
- --pipeline-parallel-size
- '1'
- --max-num-seqs
- '2'
- --max-model-len
- '1024'
- --enforce-eager
- --max-num-batched-tokens
- '256'
- --enable-chunked-prefill
- --distributed-executor-backend
- mp
- --gpu-memory-utilization
- '0.7'
- --kv-cache-memory-bytes
- '268435456'
- --no-enable-prefix-caching
- --safetensors-load-strategy
- lazy
- --load-format
- safetensors
- --quantization
- awq_marlin
- --dtype
- half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: VLLM_ADDITIONAL_ARGS
  value: --tensor-parallel-size 2 --pipeline-parallel-size 1 --max-num-seqs 2 --max-model-len
    1024 --enforce-eager --max-num-batched-tokens 256 --enable-chunked-prefill --distributed-executor-backend
    mp --gpu-memory-utilization 0.7 --kv-cache-memory-bytes 268435456 --no-enable-prefix-caching
    --safetensors-load-strategy lazy --load-format safetensors --quantization awq_marlin
    --dtype half
- name: MALLOC_ARENA_MAX
  value: '2'
- name: TOKENIZERS_PARALLELISM
  value: 'false'
- name: OMP_NUM_THREADS
  value: '1'
- name: CUDA_MODULE_LOADING
  value: LAZY
- name: VLLM_WORKER_MULTIPROC_METHOD
  value: fork
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/split-tp2-pp2/hami-mps/pp.yaml

این فایل خدمت «r2-pp» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `S-qwen7b-attempt-20260925-134248/hami-mps/rendered/pp.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --model
- /mnt/models
- --served-model-name
- Qwen/Qwen2.5-7B-Instruct-AWQ
- --host
- 0.0.0.0
- --port
- '8000'
- --tensor-parallel-size
- '1'
- --pipeline-parallel-size
- '2'
- --max-num-seqs
- '2'
- --max-model-len
- '1024'
- --enforce-eager
- --max-num-batched-tokens
- '256'
- --enable-chunked-prefill
- --distributed-executor-backend
- mp
- --gpu-memory-utilization
- '0.7'
- --kv-cache-memory-bytes
- '268435456'
- --no-enable-prefix-caching
- --safetensors-load-strategy
- lazy
- --load-format
- safetensors
- --quantization
- awq_marlin
- --dtype
- half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: VLLM_ADDITIONAL_ARGS
  value: --tensor-parallel-size 1 --pipeline-parallel-size 2 --max-num-seqs 2 --max-model-len
    1024 --enforce-eager --max-num-batched-tokens 256 --enable-chunked-prefill --distributed-executor-backend
    mp --gpu-memory-utilization 0.7 --kv-cache-memory-bytes 268435456 --no-enable-prefix-caching
    --safetensors-load-strategy lazy --load-format safetensors --quantization awq_marlin
    --dtype half
- name: MALLOC_ARENA_MAX
  value: '2'
- name: TOKENIZERS_PARALLELISM
  value: 'false'
- name: OMP_NUM_THREADS
  value: '1'
- name: CUDA_MODULE_LOADING
  value: LAZY
- name: VLLM_WORKER_MULTIPROC_METHOD
  value: fork
- name: CUDA_MPS_PIPE_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_LOG_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: '100'
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

## configurations/split-tp2-pp2/hami-mps/tp.yaml

این فایل خدمت «r2-tp» را برای مدل «Qwen/Qwen2.5-7B-Instruct-AWQ» تعریف می‌کند. منشأ: `S-qwen7b-attempt-20260925-134248/hami-mps/rendered/tp.yaml`. حالت اجرا و سناریو در نام پوشه مشخص است.

مسیر مدل، تعداد نسخه و تصویر:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
replicas: 1
image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
```

آرگومان‌های زیر به زمان اجرای خدمت فرستاده می‌شوند. طول زمینه و تعداد دنباله، ظرفیت زمینه و هم‌زمانی را محدود می‌کنند؛ درجه‌های تفکیک، نحوه تقسیم مدل را تعیین می‌کنند. گزینه بهره‌برداری حافظه و سهم حافظه HAMI دو مفهوم متفاوت‌اند؛ بدون بازآزمایی افزایش داده نشوند.

```yaml
args:
- --model
- /mnt/models
- --served-model-name
- Qwen/Qwen2.5-7B-Instruct-AWQ
- --host
- 0.0.0.0
- --port
- '8000'
- --tensor-parallel-size
- '2'
- --pipeline-parallel-size
- '1'
- --max-num-seqs
- '2'
- --max-model-len
- '1024'
- --enforce-eager
- --max-num-batched-tokens
- '256'
- --enable-chunked-prefill
- --distributed-executor-backend
- mp
- --gpu-memory-utilization
- '0.7'
- --kv-cache-memory-bytes
- '268435456'
- --no-enable-prefix-caching
- --safetensors-load-strategy
- lazy
- --load-format
- safetensors
- --quantization
- awq_marlin
- --dtype
- half
```

درخواست‌ها مبنای پذیرش زمان‌بند و حدها سقف اعلام‌شده منابع هستند. تعداد دستگاه و سهم حافظه گرافیکی با حافظه میزبان تفاوت دارند. متغیرهای محیطی حالت ترکیبی، اتصال مشتری به خدمت MPS را تعیین می‌کنند؛ آن خدمت باید جداگانه فعال و بررسی شود.

```yaml
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
env:
- name: HF_HOME
  value: /mnt/models
- name: VLLM_ADDITIONAL_ARGS
  value: --tensor-parallel-size 2 --pipeline-parallel-size 1 --max-num-seqs 2 --max-model-len
    1024 --enforce-eager --max-num-batched-tokens 256 --enable-chunked-prefill --distributed-executor-backend
    mp --gpu-memory-utilization 0.7 --kv-cache-memory-bytes 268435456 --no-enable-prefix-caching
    --safetensors-load-strategy lazy --load-format safetensors --quantization awq_marlin
    --dtype half
- name: MALLOC_ARENA_MAX
  value: '2'
- name: TOKENIZERS_PARALLELISM
  value: 'false'
- name: OMP_NUM_THREADS
  value: '1'
- name: CUDA_MODULE_LOADING
  value: LAZY
- name: VLLM_WORKER_MULTIPROC_METHOD
  value: fork
- name: CUDA_MPS_PIPE_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_LOG_DIRECTORY
  value: /tmp/nvidia-mps
- name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: '100'
```

حجم‌ها و انتخابگر گره در فایل کامل باقی مانده‌اند. در انتقال به مقصد، مسیر مشترک MPS، فضای نام، محل وزن‌ها و گره هدف را هماهنگ تغییر دهید. در پیکربندی تک‌گرهی، تأیید جای‌گذاری واقعی روی دستگاه همچنان لازم است.

# نسخه‌های تطبیق‌یافته برای گروه گره و نمایش

فایل‌های زیر تنظیم‌های زمان اجرای آزموده‌شده را با اتصال ذخیره‌سازی و انتخاب گروه گره ارائه می‌کنند. تفاوت عمدی حالت اختصاصی، سهم محاسبات ۱۰۰ برای پذیرش زمان‌بند است؛ این حالت با خط پایه تاریخی یکسان نیست. سایر آرگومان‌های زمان اجرا حفظ شده‌اند.

## pools/dedicated/10-llm.yaml

منبع اصلی: `deployment/configurations/llm-embedding/dedicated/r2-dedicated-llm.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-dedicated-llm
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://pool-dedicated-models/models/llm
nodeSelector:
  gpu-share.io/pool: dedicated
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '100'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '100'
```

## box1/dedicated/10-llm.yaml

منبع اصلی: `deployment/configurations/llm-embedding/dedicated/r2-dedicated-llm.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-dedicated-llm
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
nodeSelector:
  gpu-share.io/pool: dedicated
  kubernetes.io/hostname: mr-gholami
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '100'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '100'
```

## pools/dedicated/11-embed.yaml

منبع اصلی: `deployment/configurations/llm-embedding/dedicated/r2-dedicated-embed.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-dedicated-embed
model:
  name: BAAI/bge-m3
  uri: pvc://pool-dedicated-models/models/embed
nodeSelector:
  gpu-share.io/pool: dedicated
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '100'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '100'
```

## box1/dedicated/11-embed.yaml

منبع اصلی: `deployment/configurations/llm-embedding/dedicated/r2-dedicated-embed.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-dedicated-embed
model:
  name: BAAI/bge-m3
  uri: pvc://r2-models-pvc/models/embed
nodeSelector:
  gpu-share.io/pool: dedicated
  kubernetes.io/hostname: mr-gholami
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '100'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '100'
```

## pools/dedicated/00-storage.yaml

منبع اصلی: `round2/01-models-storage.yaml`.

این فایل حجم محلی و ادعای آن را با نام و برچسب متناظر متصل می‌کند. نام گره دارای SET باید پیش از اعمال جایگزین شود. مسیر /data2/hf-cache باید حاوی وزن‌های واقعی باشد. سیاست حفظ داده Retain باقی مانده است. این حجم به یک گره وابسته است و ذخیره‌سازی مشترک چندگرهی نیست.

## pools/hami/10-llm.yaml

منبع اصلی: `deployment/configurations/llm-embedding/hami/r2-hami-llm.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-llm
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://pool-hami-models/models/llm
nodeSelector:
  gpu-share.io/pool: hami
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
```

## box1/hami/10-llm.yaml

منبع اصلی: `deployment/configurations/llm-embedding/hami/r2-hami-llm.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-llm
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
nodeSelector:
  gpu-share.io/pool: hami
  kubernetes.io/hostname: mr-gholami
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
```

## pools/hami/11-embed.yaml

منبع اصلی: `deployment/configurations/llm-embedding/hami/r2-hami-embed.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-embed
model:
  name: BAAI/bge-m3
  uri: pvc://pool-hami-models/models/embed
nodeSelector:
  gpu-share.io/pool: hami
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
```

## box1/hami/11-embed.yaml

منبع اصلی: `deployment/configurations/llm-embedding/hami/r2-hami-embed.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-embed
model:
  name: BAAI/bge-m3
  uri: pvc://r2-models-pvc/models/embed
nodeSelector:
  gpu-share.io/pool: hami
  kubernetes.io/hostname: mr-gholami
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
```

## pools/hami/00-storage.yaml

منبع اصلی: `round2/01-models-storage.yaml`.

این فایل حجم محلی و ادعای آن را با نام و برچسب متناظر متصل می‌کند. نام گره دارای SET باید پیش از اعمال جایگزین شود. مسیر /data2/hf-cache باید حاوی وزن‌های واقعی باشد. سیاست حفظ داده Retain باقی مانده است. این حجم به یک گره وابسته است و ذخیره‌سازی مشترک چندگرهی نیست.

## pools/hami-mps/10-llm.yaml

منبع اصلی: `deployment/configurations/llm-embedding/hami-mps/r2-hami-mps-llm.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-mps-llm
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://pool-hami-mps-models/models/llm
nodeSelector:
  gpu-share.io/pool: hami-mps
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
```

## box1/hami-mps/10-llm.yaml

منبع اصلی: `deployment/configurations/llm-embedding/hami-mps/r2-hami-mps-llm.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-mps-llm
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
nodeSelector:
  gpu-share.io/pool: hami-mps
  kubernetes.io/hostname: mr-gholami
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '8192'
    nvidia.com/gpucores: '50'
```

## pools/hami-mps/11-embed.yaml

منبع اصلی: `deployment/configurations/llm-embedding/hami-mps/r2-hami-mps-embed.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-mps-embed
model:
  name: BAAI/bge-m3
  uri: pvc://pool-hami-mps-models/models/embed
nodeSelector:
  gpu-share.io/pool: hami-mps
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
```

## box1/hami-mps/11-embed.yaml

منبع اصلی: `deployment/configurations/llm-embedding/hami-mps/r2-hami-mps-embed.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-mps-embed
model:
  name: BAAI/bge-m3
  uri: pvc://r2-models-pvc/models/embed
nodeSelector:
  gpu-share.io/pool: hami-mps
  kubernetes.io/hostname: mr-gholami
resources:
  limits:
    cpu: '4'
    memory: 12Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 200m
    memory: 2Gi
    nvidia.com/gpu: '1'
    nvidia.com/gpumem: '3072'
    nvidia.com/gpucores: '50'
```

## pools/hami-mps/00-storage.yaml

منبع اصلی: `round2/01-models-storage.yaml`.

این فایل حجم محلی و ادعای آن را با نام و برچسب متناظر متصل می‌کند. نام گره دارای SET باید پیش از اعمال جایگزین شود. مسیر /data2/hf-cache باید حاوی وزن‌های واقعی باشد. سیاست حفظ داده Retain باقی مانده است. این حجم به یک گره وابسته است و ذخیره‌سازی مشترک چندگرهی نیست.

## infrastructure/mps-pool.yaml

منبع اصلی: `qwen-kserve-benchmark/20-mps-daemon.yaml`.

سرپرست اصلی MPS با همان نام نگه داشته شده است؛ تنها انتخابگر گره به گروه hami-mps و برچسب فعال‌سازی true محدود می‌شود. مسیر مشترک، شناسه کاربری، دسترسی میزبان و فرمان‌های اصلی تغییر نکرده‌اند. پیش از تغییر حالت، مشتریان تخلیه شوند.

## pools/hami/split/tp.yaml

منبع اصلی: `deployment/configurations/split-tp2-pp2/hami/tp.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-tp
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://pool-hami-models/models/llm
nodeSelector:
  gpu-share.io/pool: hami
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
```

## box1/hami/split/tp.yaml

منبع اصلی: `deployment/configurations/split-tp2-pp2/hami/tp.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-tp
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
nodeSelector:
  gpu-share.io/pool: hami
  kubernetes.io/hostname: mr-gholami
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
```

## pools/hami/split/pp.yaml

منبع اصلی: `deployment/configurations/split-tp2-pp2/hami/pp.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-pp
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://pool-hami-models/models/llm
nodeSelector:
  gpu-share.io/pool: hami
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
```

## box1/hami/split/pp.yaml

منبع اصلی: `deployment/configurations/split-tp2-pp2/hami/pp.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-pp
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
nodeSelector:
  gpu-share.io/pool: hami
  kubernetes.io/hostname: mr-gholami
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
```

## pools/hami-mps/split/tp.yaml

منبع اصلی: `deployment/configurations/split-tp2-pp2/hami-mps/tp.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-mps-tp
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://pool-hami-mps-models/models/llm
nodeSelector:
  gpu-share.io/pool: hami-mps
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
```

## box1/hami-mps/split/tp.yaml

منبع اصلی: `deployment/configurations/split-tp2-pp2/hami-mps/tp.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-mps-tp
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
nodeSelector:
  gpu-share.io/pool: hami-mps
  kubernetes.io/hostname: mr-gholami
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
```

## pools/hami-mps/split/pp.yaml

منبع اصلی: `deployment/configurations/split-tp2-pp2/hami-mps/pp.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-mps-pp
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://pool-hami-mps-models/models/llm
nodeSelector:
  gpu-share.io/pool: hami-mps
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
```

## box1/hami-mps/split/pp.yaml

منبع اصلی: `deployment/configurations/split-tp2-pp2/hami-mps/pp.yaml`.

انتخابگر زیر محل اجرای پاد را محدود می‌کند. نام خدمت مستقل است و مسیر مدل به حجم همان مقصد اشاره می‌کند. منابع و متغیرهای زمان اجرا در فایل کامل آمده‌اند.

```yaml
name: pool-hami-mps-pp
model:
  name: Qwen/Qwen2.5-7B-Instruct-AWQ
  uri: pvc://r2-models-pvc/models/llm
nodeSelector:
  gpu-share.io/pool: hami-mps
  kubernetes.io/hostname: mr-gholami
resources:
  limits:
    cpu: '2'
    memory: 6Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
  requests:
    cpu: 100m
    memory: 4Gi
    nvidia.com/gpu: '2'
    nvidia.com/gpumem: '5632'
    nvidia.com/gpucores: '50'
```

## monitoring/pool-workloads.yaml

منبع اصلی: `round2/monitoring/60-servicemonitors.yaml`.

این نسخه همان هدف پایش موجود را به‌روزرسانی می‌کند و پیشوند pool را نیز برای نام خدمت می‌پذیرد. برچسب نقش، نام گروه را نیز حفظ می‌کند تا نمودارها گروه‌های هم‌زمان را با یکدیگر جمع نکنند. هدف دوم ساخته نمی‌شود. درگاه ۸۰۰۰ و مسیر /metrics از پیکربندی اصلی حفظ شده‌اند.

## infrastructure/hami-values.yaml

مقادیر نصب تازه نسخه ۲٫۹٫۰، انتخابگر گره مدیریت‌شده، انتخابگر فضای نام، دو سهم دستگاه، ضریب حافظه و هسته برابر یک، کلاس اجرای موجود nvidia و سیاست binpack را تعریف می‌کنند. فضای نام و گره‌ها باید برچسب gpu-share.io/managed=true داشته باشند. این فایل جایگزین خودکار مقادیر نصب موجود نیست؛ قبل از ارتقا تنظیم‌های قبلی حفظ و بازبینی شوند.
