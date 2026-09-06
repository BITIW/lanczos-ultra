# lanczos-ultra

Антиалиасинговый Lanczos для RGBA8: Rust crate, C ABI и CLI. При уменьшении
поддержка фильтра расширяется пропорционально масштабу. Веса вычисляются в f64
один раз на план; проходы используют f32 без промежуточного округления и обрезки.
RGB фильтруется с premultiplied alpha, на выходе возвращается straight alpha.
Полностью прозрачные выходные пиксели имеют RGB=0, кроме точного копирования 1:1.

Цель **PSNR ≥45 дБ относительно эталонного ресайза Lanczos** проверяется
скриптом `scripts/validate.py`. Это точность воспроизведения фильтра, а не обещание
восстановить исходник после уменьшения: уменьшение необратимо теряет детали.
Результаты, методика и ограничения — в [docs/quality.md](docs/quality.md).

Режим исходного проекта **`--radius-percent 50`** восстановлен: радиус равен
50% меньшей стороны входа, без подмены числом лепестков. Исправлены антиалиасинг
и ореолы; для больших ядер при подходящих размерах используется FFT.
После подбора общих параметров режимы 50, 75 и 100 проходят сравнение
с ImageMagick Lanczos3 по PSNR, SSIM и MAE на всех трёх примерах.
Результаты и границы этого вывода — [docs/ceiling.md](docs/ceiling.md).
Исходная методика процентного режима — [docs/radius.md](docs/radius.md). По умолчанию CLI/API по-прежнему используют Lanczos;
процентный режим выбирается явно.

## Rust

В проекте-потребителе:

```toml
[dependencies]
lanczos-ultra = { path = "/path/to/lanczos-ultra", default-features = false }
```

```rust
use lanczos_ultra::{Resizer, ResizeError};

fn resize_frame(rgba: &[u8]) -> Result<Vec<u8>, ResizeError> {
    let plan = Resizer::new(1920, 1080, 960, 540)?; // Lanczos
    plan.resize(rgba)
}
```

Для потока кадров создайте `Resizer` один раз и вызывайте
`plan.resize_into(&source, &mut destination)`. План неизменяемый, его можно
разделять между потоками. `resize_rgba8(src, sw, sh, dw, dh)` — короткий вариант
однократного вызова. `Resizer::with_lobes(..., 3)` позволяет уменьшить стоимость
ценой отличия от Lanczos; контроль качества в отчёте относится к пяти лепесткам.

Буферы плотно упакованы: ровно `width * height * 4` байт, RGBA, строки сверху вниз.
Размеры должны быть ненулевыми. Stride, 16-bit, float-вход и ICC-преобразования
в этот API не входят. Обработка идёт в переданном цветовом пространстве (обычно
sRGB), без неявного перевода в linear-light. CLI декодирует изображение в RGBA8.

## .so / .dll

```sh
cargo build --release --no-default-features
```

Linux: `target/release/liblanczos_ultra.so`.
Windows (сборка на Windows): `target/release/lanczos_ultra.dll`, для MSVC также
`lanczos_ultra.dll.lib`. Экспортируется стабильный C ABI версии 1, заголовок
[include/lanczos_ultra.h](include/lanczos_ultra.h), рабочий пример
[examples/c_api.c](examples/c_api.c). Память выделяет и освобождает вызывающий код;
указатели не сохраняются. Буферы не должны пересекаться. Ошибки возвращаются
кодом; Rust panic перехватывается, но аварийное завершение аллокатора не ловится.

```sh
cc -Iinclude examples/c_api.c -Ltarget/release -llanczos_ultra \
  -Wl,-rpath,"$PWD/target/release" -o /tmp/c-api
/tmp/c-api
```

В локальном `.cargo/config.toml` установлен `target-cpu=x86-64-v3`. Для библиотеки,
распространяемой на другие x86-64 CPU, переопределите его: `RUSTFLAGS='' cargo build
--release --no-default-features` (PowerShell: `$env:RUSTFLAGS = '-C target-cpu=x86-64'`).
Для другой архитектуры используйте её target и соответствующий linker.
CI содержит сборку и C smoke test для Linux и Windows. Локально проверены Linux
.so и компиляция API для Windows GNU и MSVC; Windows runtime в этой среде отсутствует.

## CLI

```sh
cargo run --release -- input.png output.png --scale 50
cargo run --release -- input.png output.png --width 1920 --height 1080
cargo run --release -- input.png output.png --width 960 --lobes 3
```

`--scale` — целые проценты; одна сторона сохраняет пропорции с округлением до
ближайшего пикселя, минимум 1. `-H` задаёт высоту, `-h` показывает справку.
`--scale` нельзя сочетать с явными размерами. `--radius-percent 1..100` задаёт
радиус в процентах меньшей стороны входа и несовместим с `--lobes`.
Пример: `--scale 50 --radius-percent 50` уменьшает стороны вдвое, используя
радиус в половину меньшей стороны входа. Проценты выполняют разные функции.
Rust: `Resizer::with_radius_percent(sw, sh, dw, dh, 50)`.
C ABI: `lanczos_ultra_resize_radius_rgba8(..., 50)`; код ошибки 6 означает
некорректный процент. Все прежние экспортированные функции сохранены.

## Стоимость и проверка

Два разделимых прохода, параллельные строки через Rayon. Временный f32-буфер:
`dst_width * src_height * 16` байт, плюс ядра и рабочие строки Rayon.
Например, 4K → 1080p использует около 63.3 MiB промежуточного буфера.
Очень широкое увеличение при высокой исходной картинке требует больше памяти;
потоковая обработка тайлами пока не реализована. Для управления параллелизмом
задайте `RAYON_NUM_THREADS` до первого вызова или используйте собственный
`rayon::ThreadPool::install` из Rust. В C ABI план строится на каждый вызов.

```sh
cargo test --all-features
cargo test --no-default-features
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release
RAYON_NUM_THREADS=4 MAGICK_THREAD_LIMIT=4 python3 scripts/validate.py --output validation.json
RAYON_NUM_THREADS=4 cargo run --release --no-default-features --example bench
```

Скрипту сравнения нужны Python 3 и ImageMagick 7, дополнительных Python-пакетов
нет. По умолчанию он загружает Linux .so; `--library` позволяет указать DLL.
Бенчмарк исключает декодирование и сохранение изображений.