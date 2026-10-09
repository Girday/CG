# Компьютерная графика

Лабораторные работы по курсу «Компьютерная графика», МАИ.

Каждая лабораторная находится в отдельной папке со своим кодом и материалами для сдачи.

| Лабораторная | Материалы |
| --- | --- |
| [№1](lab1/) | Основы 3D-графики, вариант 7 — тор. Код, снимки и отчёт |
| [№2](lab2/) | Папка подготовлена для следующей работы |
| [№3](lab3/) | Папка подготовлена для следующей работы |
| [№4](lab4/) | Папка подготовлена для следующей работы |
| [№5](lab5/) | Папка подготовлена для следующей работы |

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

Стартовый код основан на [Vulkan Starter App](https://github.com/vladeemerr/vulkan-starter-app).
Лицензия и уведомления находятся в [LICENSE](LICENSE) и [NOTICE](NOTICE).
