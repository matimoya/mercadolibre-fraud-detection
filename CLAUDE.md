# Convenciones del proyecto

## Código

- Docstrings de una línea salvo que haya algo no obvio que explicar. Sin
  `Examples`. Los comentarios explican un *por qué*, nunca un *qué*.
- Nombres descriptivos, sin letras sueltas, también en tests y fórmulas.
- Encapsular ante repetición real, no por anticipado.
- No duplicar constantes: derivarlas de `constants.py`.
- No cambiar lo que ya funciona al refactorizar.

## Dónde va cada cosa

- `constants.py`: contrato de datos y reglas de negocio. No toca archivos.
- `paths.py`: rutas, resueltas desde la raíz del repositorio. Nunca relativas
  al directorio de trabajo.
- Resto de `src/fraud_detection/`: un módulo por responsabilidad, plano.

## Tests

- Solo lo necesario: cada test fija una regla de negocio o un bug ya pagado.
  Nada de cubrir función por función ni de inflar cobertura.
- Nombres cortos y convencionales (`test_umbral`), no oraciones.
- Datos sintéticos: la suite corre sin el CSV.
- Excluir de cobertura con motivo escrito lo que no corresponde testear.
- Verificar leyendo o corriendo el código, nunca adivinando.

## Notebooks

- Orden siempre: hipótesis → ver los datos → conclusiones.
- Cada número del informe sale de una salida de celda.
- Son narrativa: la lógica va en `src/`.
- El período reservado se toca una sola vez, con el modelo ya declarado.

## Reproducibilidad

Las semillas están fijas: el resultado tiene que ser el mismo, no parecido.
Tras tocar código, reejecutar los notebooks y diffear las salidas.

## Git

- Ramas de trabajo, PR contra `main`, conventional commits en inglés.
- Pushear solo la rama de trabajo; `main` se toca únicamente por PR.
- Sin co-autoría ni atribución, en commits ni en PRs. El PR va en español.
