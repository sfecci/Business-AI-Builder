# Alfabetización en Inteligencia Artificial: Competencias Esenciales y Pensamiento Crítico

**Curso:** Alura Latam  
**Tema:** Fundamentos de IA, productividad, Machine Learning, análisis de datos y aplicaciones prácticas  
**Tipo de proyecto:** Apuntes de estudio y resolución de actividad práctica  
**Idioma:** Español

> **Propósito:** Documentar conceptos aprendidos en el curso y aplicarlos a situaciones empresariales, con énfasis en pensamiento crítico, revisión humana y selección responsable de herramientas de IA.
>
> **Nota metodológica:** Este documento combina conceptos explicados durante las clases, ejemplos didácticos y los resultados reportados en el estudio de Dell'Acqua et al. Los ejemplos hipotéticos y las cifras basadas en índices ilustrativos se identifican como tales. Las capacidades y funciones de las herramientas pueden cambiar.

## Contenido

1. [Productividad y calidad con IA: estudio BCG](#1-productividad-y-calidad-con-ia-estudio-bcg)
2. [IA tradicional e IA generativa](#2-ia-tradicional-e-ia-generativa)
3. [Machine Learning y Deep Learning](#3-machine-learning-y-deep-learning)
4. [Familias de análisis de datos](#4-familias-de-análisis-de-datos)
5. [Herramientas y aplicaciones de IA](#5-herramientas-y-aplicaciones-de-ia)
6. [Caso práctico: SwiftBank](#6-caso-práctico-swiftbank)
7. [Colaboración humano-IA: centauros y ciborgs](#7-colaboración-humano-ia-centauros-y-ciborgs)
8. [Pensamiento crítico, seguridad y conclusiones](#8-pensamiento-crítico-seguridad-y-conclusiones)
9. [Referencias](#9-referencias)

---

## 1. Productividad y calidad con IA: estudio BCG

### Investigación de referencia

El artículo **_Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of AI on Knowledge Worker Productivity and Quality_** estudia cómo la IA generativa afecta el desempeño en tareas de consultoría. La investigación trabajó con **758 consultores de Boston Consulting Group (BCG)**.

El diseño incluyó condiciones de trabajo **sin IA**, **con GPT-4** y **con GPT-4 más capacitación/orientación sobre su uso**. Se evaluaron tareas dentro y fuera de las capacidades efectivas de la IA.

### ¿Qué es la frontera tecnológica irregular?

La *frontera tecnológica irregular* (*jagged technological frontier*) significa que la IA no realiza todas las tareas con el mismo nivel de fiabilidad. Puede resolver muy bien algunas tareas complejas y fallar en otras aparentemente similares.

- **Dentro de la frontera:** tareas donde la IA suele aportar mejoras comprobadas bajo condiciones específicas, como elaborar ideas y ciertos materiales de consultoría.
- **Fuera de la frontera:** tareas donde el modelo puede generar conclusiones erróneas y una confianza excesiva puede empeorar el desempeño humano.

> La frontera es específica de cada tarea y modelo; no equivale a afirmar que toda tarea creativa es segura o que toda tarea analítica está fuera de las capacidades de la IA.

### Resultados principales

| Indicador | Resultado reportado con IA | Interpretación |
|---|---:|---|
| Tareas completadas dentro de la frontera | **+12,2 %** | Mayor volumen de tareas completadas frente al grupo de control |
| Tiempo de ejecución dentro de la frontera | **−25,1 %** | Las tareas se realizaron más rápido |
| Calidad de los resultados dentro de la frontera | **Más de +40 %** | Mejor desempeño en las evaluaciones de calidad del estudio |
| Respuestas correctas en una tarea fuera de la frontera | **−19 puntos porcentuales** | Menor probabilidad de llegar a una solución correcta frente al grupo sin IA |

**Comparación ilustrativa con índices, no mediciones absolutas:**

| Métrica | Sin IA (índice base) | Con IA (índice relativo) |
|---|---:|---:|
| Tareas completadas | 100 | 112,2 |
| Tiempo de ejecución | 100 | 74,9 |
| Calidad evaluada | 100 | Más de 140 |

Los índices anteriores expresan cambios relativos para facilitar la lectura. **No** representan cantidades reales de tareas, minutos ni puntuaciones brutas.

### Diferencias según el desempeño inicial

En las tareas estudiadas dentro de la frontera, los participantes con desempeño inicial inferior registraron una mejora aproximada de **43 %**, frente a **17 %** entre quienes tenían un desempeño inicial superior. Esto sugiere que la asistencia de IA puede reducir algunas brechas de desempeño, sin reemplazar la formación profesional.

### Ejemplos de tareas y precauciones

| Tareas para las que la IA puede ser útil | Tareas que necesitan supervisión reforzada |
|---|---|
| Generar alternativas de producto | Resolver problemas estratégicos con evidencia contradictoria |
| Preparar borradores de correos | Evaluar riesgos legales y financieros |
| Proponer ideas de campañas | Interpretar datos de un negocio con poca contextualización |
| Organizar y resumir documentos | Recomendar decisiones con consecuencias significativas |

**Conclusión:** la IA puede mejorar productividad y calidad, pero sus respuestas deben evaluarse contra los datos, el contexto y el criterio profesional.

---

## 2. IA tradicional e IA generativa

La **IA tradicional** comprende sistemas diseñados para identificar patrones, clasificar, detectar anomalías, recomendar o predecir resultados. Puede utilizar reglas, estadística y diferentes algoritmos de aprendizaje automático.

La **IA generativa** está orientada a producir contenido nuevo —texto, imagen, audio, video o código— a partir de patrones aprendidos. Los modelos generativos actuales suelen emplear redes neuronales profundas.

| Criterio | IA tradicional | IA generativa |
|---|---|---|
| Objetivo típico | Clasificar, detectar o predecir | Generar contenido nuevo |
| Resultados | Categoría, puntuación, predicción, decisión | Texto, imagen, voz, video, código |
| Ejemplo financiero | Detectar transacciones posiblemente fraudulentas | Redactar una explicación comprensible para un cliente |
| Ejemplo comercial | Predecir abandono de clientes | Crear un borrador de correo personalizado |
| Ejemplo educativo | Identificar estudiantes con riesgo de abandono | Generar actividades y explicaciones |

### Caso aplicado: retención de clientes

1. Un modelo de **IA tradicional** analiza frecuencia de uso, reclamos e historial de interacción para calcular el riesgo de abandono.
2. Una herramienta de **IA generativa** ayuda a redactar un mensaje de retención ajustado al contexto autorizado.
3. Una persona revisa la propuesta y valida que el mensaje sea correcto, oportuno y respetuoso con la privacidad.

Los dos enfoques se **complementan**: el primero identifica una señal; el segundo puede ayudar a comunicar una respuesta.

---

## 3. Machine Learning y Deep Learning

**Machine Learning (ML)** es un subcampo de la IA en el que los modelos aprenden patrones de datos para clasificar, estimar o predecir.

**Deep Learning (DL)** es un subcampo del ML que utiliza redes neuronales de múltiples capas para aprender representaciones complejas. Por tanto, **todo Deep Learning pertenece a Machine Learning, pero no todo Machine Learning es Deep Learning**.

```text
Inteligencia Artificial (IA)
└── Machine Learning (ML)
    └── Deep Learning (DL)
```

| Característica | Machine Learning | Deep Learning |
|---|---|---|
| Enfoque | Aprendizaje de patrones mediante diversos algoritmos | Redes neuronales profundas |
| Preparación de variables | Suele utilizar variables o características seleccionadas | Puede aprender características directamente de los datos |
| Necesidad de datos | Puede funcionar con conjuntos moderados | Generalmente se beneficia de grandes volúmenes de datos |
| Cómputo | A menudo menor | A menudo mayor |
| Ejemplo | Predecir ventas usando temperatura e historial | Identificar animales en imágenes |
| Aplicaciones | Pronóstico, segmentación, clasificación | Reconocimiento de imágenes, voz y lenguaje |

### Ejemplo de la clase: ¿gato o perro?

- **ML clásico:** una persona puede diseñar o seleccionar características visuales (por ejemplo, medidas o rasgos), y el algoritmo aprende a clasificar con ellas.
- **DL:** una red neuronal puede aprender automáticamente patrones visuales relevantes a partir de muchas imágenes etiquetadas.

El resultado puede ser una **probabilidad estimada de pertenecer a cada categoría**, no una garantía de que la clasificación sea correcta.

**Aclaración:** el reconocimiento de imágenes y rostros también puede realizarse con técnicas no profundas; Deep Learning es frecuente en estas aplicaciones, pero no exclusivo.

### Otros ejemplos

- **Predicción de ventas de helados:** ML a partir de temperatura, estacionalidad e historial de ventas.
- **Autenticación facial:** procesamiento visual, habitualmente con DL, acompañado de mecanismos de seguridad adicionales.
- **Reconocimiento de voz:** modelos que convierten audio en texto.

---

## 4. Familias de análisis de datos

La elección de una herramienta debe empezar por la pregunta de negocio: **¿qué tipo de análisis se necesita?**

| Familia | Pregunta central | Objetivo | Ejemplo |
|---|---|---|---|
| **Descriptivo** | ¿Qué ocurrió? | Resumir el estado o comportamiento observado | Las ventas disminuyeron 15 % |
| **Diagnóstico** | ¿Por qué ocurrió? | Examinar causas y factores asociados | Analizar si hubo fallas o cambios de precio |
| **Predictivo** | ¿Qué podría ocurrir? | Estimar resultados futuros | Pronosticar cancelaciones |
| **Prescriptivo** | ¿Qué conviene hacer? | Comparar opciones y recomendar acciones según restricciones | Priorizar mejoras y campañas |

> El análisis descriptivo describe hechos observados. La explicación de sus causas corresponde principalmente al diagnóstico, aunque una exploración descriptiva puede motivar preguntas de causa.

### Caso didáctico: aplicación bancaria

1. **Descriptivo:** se registraron 500 reclamos durante el mes.
2. **Diagnóstico:** la revisión muestra que muchos reclamos coinciden con fallas de acceso a la aplicación.
3. **Predictivo:** un modelo estima que podrían aumentar las cancelaciones si continúan las fallas.
4. **Prescriptivo:** se comparan alternativas y se propone priorizar las correcciones, comunicar avances y medir resultados.

*Las 500 reclamaciones son una cifra hipotética usada para ilustrar los tipos de análisis.*

### Aplicaciones empresariales

- **Contabilidad y finanzas:** analizar tendencias, identificar anomalías y apoyar proyecciones.
- **Marketing:** evaluar campañas, segmentar clientes y experimentar con propuestas.
- **Producto y UX Research:** clasificar comentarios, detectar puntos de dolor (*pain points*) e identificar patrones.
- **Customer Success:** detectar riesgos de abandono y proponer medidas de retención.

No siempre es necesario realizar las cuatro familias: depende de la decisión que se desea respaldar.

---

## 5. Herramientas y aplicaciones de IA

Las siguientes herramientas fueron mencionadas durante las clases. Se indica su función principal, no una lista exhaustiva de características o planes.

| Herramienta | Categoría | Uso representativo |
|---|---|---|
| [ChatGPT](https://chatgpt.com/) | Asistente generativo | Redacción, análisis, aprendizaje y código |
| [Claude](https://claude.ai/) | Asistente generativo | Redacción, análisis documental y programación |
| [Gemini](https://gemini.google.com/) | Asistente generativo | Investigación, texto y trabajo multimodal |
| [Midjourney](https://www.midjourney.com/) | Generación visual | Creación de imágenes |
| [ElevenLabs](https://elevenlabs.io/) | Audio y voz | Narración, voces sintéticas y doblaje |
| [Runway](https://runwayml.com/) | Video | Generación y edición audiovisual |
| [Canva](https://www.canva.com/) | Diseño | Piezas gráficas con funciones de IA |
| [Gamma](https://gamma.app/) | Presentaciones | Borradores de presentaciones |
| [Lovable](https://lovable.dev/) | Creación de aplicaciones | Prototipos y aplicaciones web con instrucciones |
| [Cursor](https://cursor.com/) | Desarrollo | Asistencia para escribir y revisar código |
| [GitHub Copilot](https://github.com/features/copilot) | Desarrollo | Asistente de programación |
| [Replit](https://replit.com/) | Desarrollo | Creación y despliegue de proyectos |
| [Figma](https://www.figma.com/) | Diseño y prototipado | Interfaces y prototipos interactivos |
| [HubSpot](https://www.hubspot.com/) | CRM y marketing | Gestión de contactos y automatización |
| [Power BI](https://www.microsoft.com/en-us/power-platform/products/power-bi) | Analítica | Reportes y paneles de datos |
| [Salesforce Einstein](https://www.salesforce.com/artificial-intelligence/) | IA para CRM | Asistencia comercial y analítica |
| [Braze](https://www.braze.com/) | Marketing | Personalización de experiencias |
| [Google AI Studio](https://aistudio.google.com/) | Experimentación con modelos | Pruebas de prompts y prototipos |

### Selección de herramientas por tarea

| Necesidad | Opciones a evaluar |
|---|---|
| Borrador de correos e informes | ChatGPT, Claude, Gemini |
| Diseño visual | Canva, Midjourney |
| Video y audio | Runway, ElevenLabs |
| Presentaciones | Gamma, Canva |
| Programación y prototipos | Cursor, GitHub Copilot, Replit, Lovable, Figma |
| Analítica comercial | Power BI, HubSpot, Salesforce Einstein |

La selección debe considerar **precisión, privacidad, coste, integración, facilidad de uso y supervisión necesaria**. La existencia de una función con IA no significa que todos los planes de un producto la incluyan.

---

## 6. Caso práctico: SwiftBank

### Contexto

**SwiftBank** es una aplicación de banca digital que busca optimizar la atención al cliente mediante un asistente virtual. El reto consiste en distinguir consultas automatizables de situaciones que requieren intervención humana.

### Criterio de decisión

La complejidad de una consulta, su sensibilidad y sus consecuencias determinan el tratamiento adecuado. Las tareas repetitivas y verificables pueden gestionarse con IA cuando hay controles apropiados; las tareas ambiguas, de mayor riesgo o que requieren empatía y juicio especializado deben poder escalar a una persona.

| Consulta | Tratamiento recomendado | Consideraciones |
|---|---|---|
| Horarios o requisitos generales | Asistente virtual | Información oficial y actualizada |
| Consulta de saldo | Asistente conectado a sistemas autorizados | Autenticación y privacidad |
| Tasas vigentes | Asistente con datos verificados | Información financiera actualizada |
| Disputa de una transacción | Atención humana o equipo especializado | Investigación, fraude y derechos del cliente |
| Reclamo sensible o complejo | Atención humana | Empatía, revisión contextual y escalamiento |
| Consulta legal o financiera particular | Profesional autorizado | Validación y responsabilidad profesional |

### Alternativa correcta del ejercicio

> **Analizar la complejidad y la naturaleza de las consultas, delegando tareas repetitivas y basadas en datos a la IA, mientras que se dirigen consultas que requieren interpretación de contexto y empatía a atendientes humanos.**

Esta alternativa equilibra **eficiencia, seguridad, calidad del servicio y supervisión humana**.

### Controles recomendados

- Autenticación y autorización antes de acceder a información bancaria personal.
- Datos oficiales y actualizados para respuestas sobre tasas, productos y condiciones.
- Mecanismo claro para contactar con una persona.
- Registro de incidentes y evaluación periódica de calidad.
- Protección de datos personales y prevención de respuestas inventadas.

---

## 7. Colaboración humano-IA: centauros y ciborgs

La investigación sobre consultores también describe dos estilos de colaboración:

| Estilo | Descripción | Ejemplo |
|---|---|---|
| **Centauro** | Divide claramente las tareas entre humano e IA | La IA resume datos y la persona toma la decisión |
| **Ciborg** | Integra la IA a lo largo del flujo de trabajo, alternando continuamente | Persona e IA elaboran y revisan un documento o código paso a paso |

Ambos estilos requieren verificar resultados. No existe un estilo universalmente superior: su utilidad depende de la tarea, el riesgo y la experiencia profesional.

---

## 8. Pensamiento crítico, seguridad y conclusiones

### Buenas prácticas

1. **Definir el problema antes de elegir la herramienta.** Identificar el objetivo, la información disponible y el criterio de éxito.
2. **Dar contexto útil, pero no innecesariamente sensible.** Aportar requisitos, restricciones y formato deseado.
3. **Verificar afirmaciones y resultados.** Contrastar con fuentes fiables, documentos originales y datos del negocio.
4. **No confundir una respuesta fluida con una respuesta correcta.** La IA puede producir información inventada o desactualizada.
5. **Mantener supervisión humana proporcional al riesgo.** Especialmente en decisiones financieras, legales, médicas o de seguridad.
6. **Evaluar sesgos, privacidad y originalidad.** Los modelos pueden reproducir sesgos y generar propuestas similares a las de otros usuarios.
7. **Medir el efecto real de usar IA.** Comparar tiempos, calidad, precisión y satisfacción antes de escalar una solución.

### Conclusión general

La inteligencia artificial puede impulsar la productividad, la creatividad y el análisis de información. Sin embargo, los beneficios no son uniformes: dependen de la tarea, del modelo, del contexto y de la capacidad humana para evaluar resultados.

El concepto de **frontera tecnológica irregular** muestra por qué la supervisión humana sigue siendo esencial. Comprender las diferencias entre **IA tradicional y generativa**, **ML y DL** y las **familias de análisis** permite seleccionar herramientas más adecuadas y utilizarlas con mayor responsabilidad.

El enfoque recomendado consiste en trabajar con IA como **copiloto**, combinando automatización con pensamiento crítico y responsabilidad profesional.

---

## 9. Referencias

1. Dell'Acqua, F., et al. (2023). **_Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of AI on Knowledge Worker Productivity and Quality_.** SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4573321
2. **Alura Latam.** Curso *Alfabetización en Inteligencia Artificial: Competencias Esenciales y Pensamiento Crítico*. Transcripciones y actividad práctica proporcionadas para estos apuntes.

---

## Uso de este repositorio

Este `README.md` puede utilizarse como material de repaso y como registro de aprendizaje. Las tablas y ejemplos están escritos con fines educativos; no deben interpretarse como asesoramiento financiero, legal o como validación técnica de un producto de IA.

**Temas clave para seguir estudiando:** prompting, evaluación de respuestas, métricas de calidad, sesgos, seguridad, privacidad y agentes de IA.
