**GOBERNANZA**

En el contexto de los proyectos de MLOps, la gobernanza desempeña un papel crucial para asegurar que los modelos de aprendizaje automático no solo funcionen correctamente, sino que también operen de manera ética, responsable y alineada con los objetivos del negocio. En este proyecto, cuyo objetivo es predecir la emoción que transmite una canción en función de sus características, hemos adoptado un enfoque de gobernanza para garantizar que el modelo sea transparente, seguro y ajustado a los principios éticos.

Para establecer una base sólida en la gobernanza de este sistema de MLOps, nos inspiramos en los principios propuestos por Treveil (2020), quien define ocho pasos clave para estructurar la gobernanza en proyectos de aprendizaje automático:

1\.  Comprender y clasificar los casos de uso.  
2\.  Establecer una posición ética.  
3\.  Establecer responsabilidades.  
4\.  Determinar las políticas de gobernanza.  
5\.  Integrar políticas en el proceso MLOps.  
6\.  Selección de herramientas de gestión centralizada para el gobierno.  
7\.  Involucrar y educar.  
8\.  Monitorear y refinar.

Al combinar estos principios con un proceso de MLOps estructurado, no solo hemos establecido una guía de acción, sino también un proceso de gobernanza integral. Este enfoque garantiza que el modelo pueda predecir de manera precisa y ética la emoción transmitida por una canción, proporcionando un valor confiable y responsable tanto para los desarrolladores como para los usuarios finales.

**1\. CASOS DE USO**

Se identificaron tres casos de uso principales en el proyecto:

**1.3 Cumplimiento de regulaciones**: 

* Evaluamos los aspectos regulatorios locales y regionales. Dado que el modelo solo predice emociones musicales en un conjunto de datos y no maneja datos sensibles, no se identificaron desviaciones en términos de leyes de privacidad.


**1.2 Pruebas de código y disponibilidad**: 

* Realizamos pruebas tanto a nivel de objetos como de integración para validar la estabilidad y consistencia del modelo. Estos experimentos se enfocaron en los indicadores de desempeño y en pruebas de punta a punta.


**1.3 Pruebas en preproducción**: 

* Utilizamos una infraestructura en GitHub (código), mlflow (almacenamiento de modelos) y DVC (almacenamiento de datos). El pase a producción incluye verificar código, manuales, tipo de modelo, métricas, y documentación en el formato adecuado, realizando una prueba inicial para validar la precisión de los resultados.

Estos casos de uso permiten monitorear la vida útil del modelo y ajustarlo en caso de que se ingresen nuevos datos o cambien las observaciones.

**2\. ÉTICA, PRINCIPIOS Y COMPROMISOS**

El uso de herramientas de Inteligencia artificial y aprendizaje automático trae consigo riesgos y áreas de oportunidad. Para reducir los riesgos asociados al uso de IA, se establecen los siguientes principios:

**2.1 Equidad**

*   **Riesgos:** Restringir las herramientas o la información a un grupo reducido de personas puede desencadenar un desbalance de poder y limitar el acceso equitativo a tecnologías innovadoras.  
*  **Compromiso:** Democratizar el acceso a las herramientas y la información, haciéndolas disponibles para el público en general. Nos comprometemos a diseñar nuestras soluciones de manera inclusiva, minimizando los sesgos para promover una experiencia justa para todos los usuarios.

**2.2 Confiabilidad**

* **Riesgos:** La IA podría realizar predicciones imprecisas, lo que podría afectar la experiencia del usuario y reducir la confianza en el sistema.  
* **Compromiso:** Diseñar, entrenar y actualizar continuamente nuestros modelos para asegurar que las predicciones sean precisas y consistentes. Nos comprometemos a validar nuestros resultados y mejorar el sistema con base en evidencia científica y retroalimentación de los usuarios, buscando un desempeño de alta calidad en todo momento.

**2.3 Seguridad**

* **Riesgos:**El uso inadecuado de la IA podría provocar vulnerabilidades que pongan en riesgo la seguridad de los datos de los usuarios o el sistema en su conjunto.  
* **Compromiso:** Implementar medidas de seguridad robustas en todas las etapas del desarrollo de la IA, protegiendo los datos de nuestros usuarios mediante encriptación y protocolos de seguridad avanzados. Nos comprometemos a monitorear activamente la seguridad de nuestra tecnología para prevenir accesos no autorizados y responder rápidamente ante cualquier amenaza.

**2.4 Transparencia**

* **Riesgos:** La falta de claridad en el funcionamiento de la IA podría hacer que los usuarios duden de sus predicciones o no comprendan el valor real del sistema.  
* **Compromiso:** Proveer explicaciones claras y accesibles sobre cómo nuestra IA analiza y predice emociones a partir de la música. Nos comprometemos a ser transparentes sobre los datos utilizados y los métodos implementados, así como a informar a los usuarios sobre el alcance y las limitaciones de nuestro sistema.

**2.5 Privacidad**

* **Riesgos:** La filtración de información privada o confidencial puede comprometer la privacidad y seguridad del usuario, además de generar desconfianza en la IA.  
* **Compromiso:**Respetar la privacidad de nuestros usuarios utilizando exclusivamente sus datos de forma anónima y agrupada. Nuestro sistema no analiza información personal de manera individual, sino que se enfoca en metadatos y estadísticas grupales para mejorar el desempeño del modelo y la experiencia del usuario sin comprometer su privacidad.

**2.6 Responsabilidad**

* **Riesgos:** Un uso irresponsable de la IA podría llevar a la desinformación o al impacto negativo en las decisiones de los usuarios.  
* **Compromiso:** Asumir la responsabilidad sobre el impacto de nuestras predicciones y garantizar que nuestros modelos se desarrollen y utilicen con ética y profesionalismo. Nos comprometemos a mejorar continuamente nuestra IA, y a ofrecer apoyo a los usuarios para que comprendan el uso correcto del sistema, promoviendo siempre un uso seguro y ético de la tecnología.

A partir de estos principios que otros autores también los mencionan como Treveil y otros, mencionan que los principio deben incluir medidas regulatorias, la responsabilidad de la IA y un marco de gobernanza sólido (2020).

**3\. RESPONSABILIDADES**

La gobernanza en MLOps es el producto de revisar, auditar, enseñar y trabajar en la continuidad de los modelos implementados o que se están fabricando. El acompañamiento de los diferentes roles determinan el grado de acompañamiento, pero además el grado de responsabilidades, las organizaciones tienen una estructura jerárquica, es por ello además que debemos involucrar a todo la organización, para que todos entiendan no el modelo, sino que puede o no puede hacer el modelo, y las implicaciones de no hacer o predecir un resultado de forma incorrecta.

Por lo tanto la responsabilidad es compartida, en la siguiente matriz RACI podemos observar los roles y responsabilidades de cada persona de la organización.

Dentro de la matriz RACI hay 3 roles:

- Ejecutor      (E)  
- Revisor       (R)  
- Informador  (I).

| Tareas | Experto del negocio | Ingeniero de datos | Arquitecto de ML | Científico de datos | Gobierno  | Ingeniero de software | DevOps |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Identificar el objetivo del modelo | E | E | E | E |  |  |  |
| Preparación de datos | I | E | R | E | I |  |  |
| Modelado |  | E | I | E | I |  |  |
| Aceptación de los datos de prueba,  | I | R | R | E | I |  |  |
| Aceptación del modelo | I | R | I | E | R |  | I |
| Pruebas (unitarias y de integración) | I | I |  | E | I | I | I |
| Aceptación de las pruebas | R | I | E/R | R | E/R | R | E |
| Ambientación en interfaz de usuario  | R | I | I | R | R | E | I |
| Aceptación de la documentación y código |  | I |  | I | E/R |  | I |
| Pase a producción | I | I | I | I | R | I | E |
| Monitoreo | I | I |  | E | R | I | E |

**4\. POLÍTICAS DE GOBERNANZA**

Las normas ISO e IEEE ofrecen un marco de referencia invaluable para abordar los desafíos planteados en este proyecto. A continuación, se presentan algunas de las normas más relevantes:

**4.1 Normas ISO**

* **ISO/IEC 25010:2011** \- Modelo de Calidad del Software  
  * **Características de calidad:** Esta norma define un conjunto de características que permiten evaluar la calidad de un producto de software. En el contexto de MFLOps, estas características se aplican a los modelos de machine learning y a las plataformas que los soportan.  
  * **Conceptos clave:** Funcionalidad, rendimiento, compatibilidad, usabilidad, confiabilidad, seguridad, mantenibilidad y portabilidad.  
  * **Prácticas clave:**  
    * **Definición clara de requisitos:** Establecer de manera precisa los requisitos de calidad para cada modelo y componente del sistema.  
    * **Evaluación continua:** Implementar mecanismos para medir y evaluar continuamente las características de calidad de los modelos y plataformas.  
    * **Mejora continua:** Utilizar los resultados de las evaluaciones para identificar áreas de mejora y realizar ajustes en los procesos y modelos.  
        
* **ISO/IEC 27001:2013** \- Sistema de Gestión de Seguridad de la Información (SGSI)  
  * **Enfoque sistemático:** Esta norma proporciona un marco para establecer, implementar, mantener y mejorar continuamente un SGSI.  
  * **Conceptos clave:** Confidencialidad, integridad y disponibilidad de la información.  
  * **Prácticas clave:**  
    * **Análisis de riesgos:** Identificar, evaluar y tratar los riesgos para la seguridad de la información.  
    * **Controles de seguridad:** Implementar controles técnicos, administrativos y físicos para proteger la información.  
    * **Gestión de incidentes:** Establecer procedimientos para la detección, respuesta y recuperación ante incidentes de seguridad.  
        
* **ISO/IEC 27005:2022** \- Gestión de Riesgos de Seguridad de la Información  
  * **Ciclo de vida de la gestión de riesgos:** Define un proceso estructurado para identificar, evaluar y tratar los riesgos de seguridad de la información.  
  * **Conceptos clave:** Activo, amenaza, vulnerabilidad, impacto y riesgo.  
  * **Prácticas clave:**  
    *  **Inventario de activos:** Identificar y catalogar los activos de información críticos.  
    * **Análisis de amenazas y vulnerabilidades:** Evaluar las posibles amenazas y vulnerabilidades que podrían afectar a los activos.  
    * **Evaluación de riesgos:** Calcular el nivel de riesgo asociado a cada amenaza y vulnerabilidad.


* **ISO/IEC 30143:2018** \- Gestión de Riesgos de Seguridad de los Activos  
  * **Enfoque en los activos:** Se centra en la gestión de riesgos asociados a activos específicos, como los modelos de machine learning.  
  * **Conceptos clave:** Valor del activo, exposición al riesgo, controles de seguridad y tolerancia al riesgo.  
  * **Prácticas clave:**  
    * **Clasificación de activos:** Clasificar los activos según su valor e importancia para la organización.  
    * **Evaluación de la exposición al riesgo:** Evaluar la probabilidad y el impacto de las amenazas a los activos.  
    * **Definición de controles:** Implementar controles de seguridad específicos para proteger los activos.


* **IEEE Std 15288-2008** \- Modelo de Proceso de Ingeniería de Sistemas  
  * **Enfoque en sistemas complejos:** Proporciona un marco para la gestión de proyectos de ingeniería de sistemas complejos, como los sistemas de MLOps.  
  * **Conceptos clave:** Ciclo de vida del sistema, requisitos, diseño, verificación y validación.  
  *  **Prácticas clave:**  
    * **Gestión de requisitos:** Definir, gestionar y verificar los requisitos del sistema.  
    * **Arquitectura de sistemas:** Diseñar la arquitectura del sistema y sus componentes.  
    * **Integración y verificación:** Integrar los componentes del sistema y verificar que cumplan con los requisitos.


* **IEEE Std 16326-2009** \- Ingeniería de Sistemas Basados en Modelos  
  * **Modelado de sistemas:** Define un vocabulario y conceptos comunes para el modelado de sistemas complejos.  
  * **Conceptos clave:** Modelo, metamodelo, simulación y análisis.  
  * **Prácticas clave:**  
    * **Desarrollo de modelos:** Crear modelos formales para representar el sistema y sus componentes.  
    * **Simulación y análisis:** Utilizar la simulación para evaluar el comportamiento del sistema y realizar análisis.

**5\. INTEGRACIÓN DE POLÍTICAS AL PROYECTO**

### **5.1 Política de Seguridad de Datos**

* **Protección de Datos**: Establece controles estrictos para proteger los datos sensibles de usuarios y clientes, cumpliendo con regulaciones como GDPR y CCPA. Incluye medidas de cifrado, acceso restringido y almacenamiento seguro.  
* **Gestión de Permisos**: Define roles y permisos para garantizar que solo el personal autorizado acceda a datos específicos. Este enfoque minimiza el riesgo de brechas de seguridad y pérdida de datos.

### **5.2 Política de Privacidad**

* **Consentimiento y Transparencia**: Informa a los usuarios sobre cómo se recopilan, utilizan y almacenan sus datos, obteniendo su consentimiento explícito. Esto crea confianza y asegura el cumplimiento de regulaciones de privacidad.  
* **Acceso y Eliminación de Datos**: Proporciona a los usuarios opciones para acceder a sus datos o solicitar su eliminación en cualquier momento. La política promueve la transparencia y fortalece la relación con los usuarios.

### **5.3 Política de Ética en el Uso de IA**

* **Evitar Sesgos y Discriminación**: Implementa revisiones periódicas del modelo para reducir sesgos en las predicciones y decisiones automatizadas. Se enfoca en un uso ético de la IA que beneficie a todos los usuarios de manera equitativa.  
* **Transparencia Algorítmica**: Documenta y comunica cómo funciona el modelo para que los usuarios comprendan los criterios de las decisiones. Aumenta la confianza en el sistema al mostrar claridad en los procesos de IA.

### **5.4 Política de Uso Responsable**

* **Condiciones de Uso para Usuarios**: Establece reglas claras sobre el uso permitido del sistema, incluyendo restricciones para prevenir el abuso. Define penalizaciones para usuarios que incumplan los términos.  
* **Detección de Abusos**: Monitorea el sistema en busca de comportamientos irregulares, con alertas para posibles usos indebidos. Esta vigilancia garantiza la integridad y el uso seguro del proyecto.

### **5.5 Política de Monitoreo y Actualización de Políticas**

* **Revisión y Ajuste Continuo**: Establece revisiones periódicas de las políticas para adaptarlas a cambios en el contexto regulatorio o tecnológico. Esto asegura que el proyecto cumpla con las normativas actuales y emergentes.  
* **Comunicación de Cambios**: Los usuarios y el equipo son informados de cualquier modificación en las políticas para mantener la transparencia. Garantiza que todos los involucrados estén al tanto de las normas vigentes.

# **6\. HERRAMIENTAS**

Las herramientas seleccionadas son fundamentales dentro de la metodología MLOps para asegurar un flujo de trabajo ágil, colaborativo y trazable. Cada una de estas herramientas es referente en el mercado y se integra para optimizar el desarrollo, implementación y mantenimiento del modelo.

* **DVC (Data Version Control)**  
  DVC permite gestionar versiones de los datos, código y modelos, integrando un sistema de versionamiento específico para datos de entrenamiento, además de habilitar el almacenamiento de logs y seguimiento de cambios en colaboración con MLflow. Esto facilita la reproducibilidad y trazabilidad del proyecto, permitiendo retroceder a versiones previas de los datos y experimentos.  
* **Docker**  
  Docker permite la virtualización de ambientes de desarrollo y despliegue, encapsulando MLflow, DVC y otros entornos de prueba en contenedores. Esto asegura la consistencia en los ambientes de trabajo independientemente de la máquina, lo cual facilita tanto el desarrollo colaborativo como la implementación en producción.  
* **MLflow**  
  MLflow es utilizado para el seguimiento de experimentos, la gestión de modelos y el registro de métricas y artefactos. Al integrar MLflow en el proyecto, se centralizan los experimentos y sus resultados, permitiendo comparar versiones de modelos, registrar métricas y seleccionar el mejor modelo basado en resultados cuantitativos.  
* **GitHub**  
  GitHub se usa como sistema de control de versiones y permite el trabajo en equipo bajo la metodología de ramas. La estructura de control de versiones sigue la matriz RACI para asignar claramente las responsabilidades y facilita la colaboración, la revisión de código y la integración continua.  
* **Visual Studio Code (VSCode)**  
  VSCode es el IDE utilizado para el desarrollo del proyecto, configurado con Python 3.10 y todas las dependencias necesarias, incluyendo librerías específicas para el modelo de machine learning y módulos auxiliares. Su integración con GitHub y Docker permite una experiencia de desarrollo fluida y la fácil configuración de entornos de trabajo compartidos.

Cada herramienta ha sido seleccionada para cubrir aspectos específicos de la gobernanza en el proyecto, asegurando control, trazabilidad y eficiencia a lo largo del ciclo de vida del modelo.

**7\. ACERCAMIENTO Y EDUCACIÓN**

Para asegurar que los involucrados en el proyecto comprendan y se comprometan plenamente con su desarrollo, se establece un proceso de documentación y educación continua. Este enfoque permite que tanto el equipo de desarrollo como los usuarios finales y otros interesados comprendan el propósito, funcionamiento y objetivos del proyecto. Las siguientes herramientas y recursos se utilizan para facilitar este entendimiento:

### **7.1 README.md:** Explicación General del Proyecto

* **Resumen y Propósito**: El archivo `README.md` proporciona una descripción general del proyecto, incluyendo sus objetivos, público objetivo y contexto de aplicación. Esto permite a los nuevos integrantes o interesados comprender rápidamente de qué se trata el proyecto y su relevancia.  
* **Estructura del Proyecto**: Incluye un desglose de la estructura de carpetas y archivos, explicando las funciones de cada componente clave. Esto ayuda a orientar al equipo y facilita la navegación por el código y los recursos relacionados.

  ### **7.2 Configuración.yml:** Configuración Detallada

* **Configuración de Entorno**: El archivo `configuración.yml` describe en detalle los requisitos y las configuraciones necesarias para ejecutar el proyecto. Esto incluye versiones de software, dependencias y variables de entorno, lo cual asegura que todos los usuarios puedan replicar y ejecutar el proyecto sin inconvenientes.  
* **Personalización y Adaptación**: También se documentan opciones de personalización y parámetros que pueden ajustarse según el contexto o necesidades específicas del usuario. Esto facilita la adaptación del proyecto a diferentes entornos y permite un uso más flexible.

  ### **7.3 Comentarios y Docstrings en el Código**

* **Documentación Interna**: Los comentarios y `docstrings` en el código explican la funcionalidad de cada función, clase y bloque de código. Estos comentarios sirven como guía para los desarrolladores actuales y futuros, facilitando la comprensión y el mantenimiento del proyecto.  
* **Estandarización de Documentación**: Se siguen convenciones de estilo para garantizar que los `docstrings` sean uniformes y claros, proporcionando detalles de entradas, salidas, excepciones y lógica clave. Esto contribuye a que el código sea autodescriptivo y fácil de entender.

  ### **7.4 Proyecto de Ejemplo: Análisis de Emociones en Música Turca**

* **Demostración Práctica**: Se incluye un proyecto práctico titulado "ML de Emociones en Música Turca" como un caso de uso para ilustrar el funcionamiento del modelo y la relevancia del análisis emocional en la música. Este ejemplo práctico permite a los interesados visualizar los resultados y comprender cómo se aplican los principios del proyecto en un contexto real.  
* **Acceso a la Demostración**: La demostración está disponible en un servidor accesible desde [http://24.144.69.175:5050](http://24.144.69.175:5050), permitiendo a los usuarios interactuar con el proyecto de manera práctica. Este acceso permite que los usuarios y el equipo puedan experimentar el proyecto de primera mano, comprender sus capacidades y sugerir mejoras.

  ### **7.5 Sesiones de Capacitación y Talleres**

* **Capacitación para el Equipo**: Se organizan sesiones de capacitación y talleres para el equipo y otros interesados en el proyecto. Estos eventos cubren temas como la configuración del entorno, el flujo de trabajo, las buenas prácticas de desarrollo y el uso de las herramientas documentadas.  
* **Educación para Usuarios Finales**: Se diseñan sesiones educativas orientadas a los usuarios finales para enseñarles cómo interpretar los resultados y aprovechar las funcionalidades del sistema. La capacitación ayuda a asegurar que los usuarios puedan utilizar el proyecto de manera efectiva y con confianza.

  ### **7.6 Revisión y Actualización Continua de Documentación**

* **Actualización Periódica**: La documentación es revisada y actualizada de forma regular para reflejar cambios en el código, configuraciones y funcionalidades del proyecto. Esto asegura que toda la información esté al día y que los nuevos miembros puedan integrarse fácilmente.  
* **Retroalimentación de Usuarios**: Se alienta a los usuarios y al equipo a proporcionar retroalimentación sobre la claridad y utilidad de la documentación. Las sugerencias se consideran en cada actualización para mejorar continuamente la calidad de la documentación.

**8\. MONITOREO Y REFINAMIENTO**  
El monitoreo y la mejora continua de los modelos de machine learning son fundamentales para garantizar su rendimiento, seguridad y precisión a lo largo del tiempo. Este proceso incluye la implementación de pruebas unitarias y de integración, el monitoreo de métricas de desempeño y la actualización de los modelos según los cambios en los datos o en el entorno del sistema. Estos pasos se detallan a continuación.

### **8.1 Pruebas Continuas**

* **Pruebas Unitarias**: Validan cada componente del modelo para asegurar que funcione de acuerdo a los requisitos específicos. Se recomienda ejecutarlas con cada cambio en el código para detectar errores rápidamente.  
* **Pruebas de Integración**: Garantizan que los componentes del sistema interactúen sin problemas, previniendo errores en producción. Son esenciales antes de cada despliegue y deben realizarse al menos semanalmente.

### **8.2 Métricas de Rendimiento y Calidad del Modelo**

* **Métricas de Desempeño**: Se monitorean métricas como precisión, recall, F1 score, y matriz de confusión para asegurar la calidad de las predicciones. Estas métricas se registran continuamente para detectar caídas en el rendimiento.  
* **Métricas de Consumo de Recursos**: Miden el impacto del modelo en tiempo de latencia, memoria y uso de CPU/GPU. Ayudan a optimizar recursos y a prevenir problemas en el rendimiento del sistema.

### **8.3 Monitoreo del Desempeño del Modelo en Producción**

* **Detección de Deriva del Modelo**: Monitorea cambios en los datos y en las métricas para identificar cuando el modelo ya no es fiable. Una vez detectada una desviación significativa, se activa el reentrenamiento.  
* **Supervisión de Desviaciones**: Detecta comportamientos inusuales en el modelo y activa alertas cuando el desempeño se desvía de los parámetros definidos. Esto previene problemas serios al intervenir antes de que afecten al usuario final.

### **8.4 Sistema de Aprendizaje y Mejora Continua**

* **Pipeline de Aprendizaje Continuo**: Automatiza la actualización del modelo incorporando nuevos datos y re-entrenamientos regulares. Esto asegura que el modelo se mantenga relevante con el tiempo.  
* **Validación en Entornos de Prueba**: Realiza pruebas A/B y compara el modelo actualizado con el anterior antes de implementarlo. Esto garantiza mejoras sin riesgos de caída en producción.

### **8.5 Documentación de Actualizaciones y Mejora del Sistema**

* **Registro de Cambios**: Documenta todas las actualizaciones de modelo y datos para mantener un historial claro. Asegura que el equipo tenga acceso a información sobre cada ajuste y sus efectos.  
* **Actualización de Documentación y Capacitación**: Se mantiene la documentación al día para reflejar los cambios recientes, y el equipo recibe capacitación sobre el uso óptimo de las nuevas versiones. Esto mejora el uso y entendimiento del modelo actualizado.

