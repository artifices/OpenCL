#include "CL/cl.hpp"
#include <iostream>
#include <string>

// Esta constante aca esta solo para que se calcule el largo para luego declarar el tamano exacto que se necesita
// para el buffer
const std::string hw = "Hello World";


// Un kernel en OpenCL viene a ser una funcion paralelizable que cumple un rol especial.
// Para correr un kernel se necesita una plataforma, un contexto (vendria a manejar todo), un dispositivo (CPU/GPU), un
// programa y una cola de comandos. Todos estos se preparan para poder ejecutar un kernel.
// Curiosamente un kernel se declara como un string...Podria ser importado de un .txt aca esta hardcoded
// El kernel esta separado del programa.
std::string prog(
	"#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n" 
	"__constant char hw[] = \"Hello World\"; \n"
	"__kernel void hello(__global char * out) {\n" 
	"  size_t tid = get_global_id(0); \n" 
	"  out[tid] = hw[tid]; \n" 
	"}\n");


// Aqui comienza el programa
int main(void) {
	// Es solo para ir guardando los errores, pero no hago nada con ellos
	cl_int err;
	
	// Obtener la lista de plataformas de OpenCL que estan disponibles (una plataforma seria un SDK de Intel, o un SDK de 
	// AMD, o un SDK de nVidia), se podrian tener varios, en nuestro caso el de AMD. Se crea un vector de 
	// clase Plataforma para almacenar las plataformas luego de que el metodo
	// get las obtenga. A partir de la primera plataforma se crea un contexto. El contexto tiene dispositivos (CPUs/GPUs, 
	// memoria, programas, cola de comandos que permiten ejecutar los kernels. Todo se hace a partir del contexto y
	// como que se van guardando las "opciones" en el contexto
	std::vector<cl::Platform> platformList;
	cl::Platform::get(&platformList);	
	cl_context_properties cprops[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};
	cl::Context context( CL_DEVICE_TYPE_CPU, cprops, NULL, NULL, &err); // Podria ser CL_DEVICE_TYPE_GPU para usar GPU

	// Se crea un buffer de tamano del la palabra "Hello world" para almacenar el resultado del 
	// kernel hello que ejecuta el dispositivo. Aca se crea con espacio de mas, 100 bits. Este buffer se va comunicar
	// con el contexto creado anteriormente. Con la flag "CL_MEM_USE_HOST_PTR" se esta pidiendo que OpenCL usa
	// la memoria directamente. El buffer viene a ser la memoria que se va ultilizar para lo que ser quiera hacer.
	size_t mem_size = 100;
	char * salida = new char[hw.length()+1];
	cl::Buffer buffer( context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, hw.length()+1, salida, &err);

	// Obtener los dispositivos del contexto actual, por ahora como esta instanciado seria solo en el CPU
	// La informacion sobre los dispositivos se almacena un vector tambien
	std::vector<cl::Device> devices;
	devices = context.getInfo<CL_CONTEXT_DEVICES>();	

	// Dar el codigo de prog (el que esta arriba, que encapsula al kernel) a OpenCL. Este prog va ser 
	// ejecutado por los dispositivo/s, primero aca se crea un source (fuente) a partir del string de arriba
	cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length()+1));

	// Aca se compila y se "construye" el programa (a partir del source que se creo arriba) con los dispositivos
	cl::Program program(context, source);
	err = program.build(devices,"");

	// Un programa puede tener mucho puntos de entrada y a estos se les llama kernels. Para llamar a un kernel se
	// tiene que crear un objeto de tipo kernel. Basicamente hay que decirle que vamos usar "hello", y lo asociamos
	// al programa que se instancio arriba
	cl::Kernel kernel(program, "hello", &err);
	err = kernel.setArg(0, buffer);

	// Todas las computaciones de los dispositivos (calculos, acciones etc.) se hacen a partir de una cola de comandos.
	// La cola de comandos tiene un mapeo (relacion) de uno a uno con un dispositivo.
	// La cola se crea a partir de un contexto asociado. Luego a la cola creada se le pueden ir encolando kernels
	// para ejecutarse en el dispositivo asociado.

	// El numero total de elementos tamano se llama trabajo global (global work size). Elementos individuales se 
	// llaman work-items y estos pueden ser agrupados en work-groups. En este ejemplo cada "work-item" computa
	// una letra del string "Hello world". EL ultimo parametro del enqueueNDRangeKernel event, se puede utilizar
	// para consultar el estado del comando al cual esta asociado
	cl::CommandQueue queue(context, devices[0], 0, &err);
	cl::Event event;
	err = queue.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange(hw.length()+1), cl::NDRange(1, 1), NULL, &event);

	// Se usa el objeto event para bloquear el proceso hasta que este completo
	// Sin esto muestra cualquier cosa, porque esto asegura que el kernel entero se termino de ejecutar antes de que
	// se devuelva el resultado al buffer. Es una forma de "lock" o bloqueo para esperar que se termine de ejecutar
	// todo lo concurrente
	event.wait();

	// Se lee lo que esta almacenado en el buffer y se guarda en la variable outH
	err = queue.enqueueReadBuffer(buffer, CL_TRUE, 0, hw.length()+1, salida);

	// Se muestra simplemente lo que esta adentro de salida
	std::cout << salida << std::endl;
	
	system("pause");
}