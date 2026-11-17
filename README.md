# Компьютерная графика

Лабораторные работы по курсу «Компьютерная графика», МАИ.

Каждая лабораторная находится в отдельной папке со своим кодом и материалами для сдачи.

| Лабораторная | Материалы |
| --- | --- |
| [№1](lab1/) | Основы 3D-графики, вариант 7 — тор. Код, снимки и отчёт |
| [№2](lab2/) | Камера и освещение Блинн–Фонга: направленный, точечные и прожекторные источники |
| [№3](lab3/) | Текстуры, dynamic UBO, анимированное сэмплирование, specular и emissive карты |
| [№4](lab4/) | Карты теней от солнца и прожектора через Vulkan Dynamic Rendering |

Для сдачи первой работы: [lab1](https://github.com/Girday/CG/tree/main/lab1).

Сборка и запуск описаны в [lab1/README.md](lab1/README.md).
Также можно собрать первую лабораторную из корня репозитория:

```sh
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel
ctest --test-dir build-debug --output-on-failure
cd lab1
../build-debug/lab1/cg-lab1
```

Для запуска второй работы на macOS:

```sh
./lab2/run-macos.sh
```

Параметры, теория и общие команды сборки: [lab2/README.md](lab2/README.md).

Для запуска третьей работы на macOS:

```sh
./lab3/run-macos.sh
```

Материалы, эффекты и устройство текстурирования: [lab3/README.md](lab3/README.md).

Четвёртая работа запускается командой `./lab4/run-macos.sh`.
Управление тенями, устройство проходов и снимки сцены описаны в [lab4/README.md](lab4/README.md).

Стартовый код основан на [Vulkan Starter App](https://github.com/vladeemerr/vulkan-starter-app).
Лицензия и уведомления находятся в [LICENSE](LICENSE) и [NOTICE](NOTICE).
