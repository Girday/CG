# Лабораторная работа №1 — Основы 3D-графики

Вариант 7: тор. Реализация на C++20 с Vulkan, GLFW и ImGui.

Выполнены переключение перспективной и ортографической проекций, управление положением,
поворотом и масштабом, пространственная анимация с паузой и параметрами траектории,
выбор цвета, процедурные цвета вершин и отрисовка двух объектов с отдельными наборами дескрипторов.

## Материалы

- [Отчёт PDF](reports/М8О-301БВ-24_Григорьев_Лаб1.pdf)
- [Отчёт DOCX](reports/М8О-301БВ-24_Григорьев_Лаб1.docx)
- [Исходный код](source/)
- [Шейдеры](shaders/)
- [Перспективная проекция](docs/lab1-perspective.png)
- [Ортографическая проекция](docs/lab1-orthographic.png)

## Сборка и запуск

Нужны компилятор C++20, CMake 3.20+ и Vulkan SDK.
Команда `glslc` должна быть доступна в PATH. На macOS предварительно подключите
окружение установленного SDK через его `setup-env.sh`; Vulkan работает через MoltenVK.
CMake скачивает GLFW, vk-bootstrap, Vulkan Memory Allocator и ImGui.

Из этой папки на macOS или Linux:

```sh
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel
ctest --test-dir build-debug --output-on-failure
./build-debug/cg-lab1
```

Для Windows с Visual Studio:

```powershell
cmake --preset msvc-debug
cmake --build build-debug --config Debug --parallel
ctest --test-dir build-debug -C Debug --output-on-failure
.\build-debug\Debug\cg-lab1.exe
```

Рабочая директория при запуске — `lab1`, чтобы приложение находило файлы в `shaders/`.
