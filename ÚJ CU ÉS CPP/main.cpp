//Main.cpp - Mostly openGL, contains the main()
//simulation methods are called from the openGL callbacks

#include "kernel.h"


/////Variables for controlling the simulation, these are changed by the user input

//Force
float2 force;

//Visualization Method
int visualizationMethod = 0;

//Visualization (Density) color
float4 densityColor;

//Width and height of the window
int width = 512;
int height = 512;

//// OpenGL - common openGL callback functions, and handling of user inputs
int method = 1;
bool keysPressed[256];

//Plain openGL initialization
void initOpenGL() {
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
	}
	else {
		if (GLEW_VERSION_3_0)
		{
			std::cout << "Driver supports OpenGL 3.0\nDetails:" << std::endl;
			std::cout << "  Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
			std::cout << "  Vendor: " << glGetString(GL_VENDOR) << std::endl;
			std::cout << "  Renderer: " << glGetString(GL_RENDERER) << std::endl;
			std::cout << "  Version: " << glGetString(GL_VERSION) << std::endl;
			std::cout << "  GLSL: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
		}
	}

	glClearColor(0.17f, 0.4f, 0.6f, 1.0f);
}

//The simulation is iterated, and the visualization is calculated when the screen is being redrawn
int i = 0;
void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	simulationStep();
	if (i++ % 3 == 0) {
		//addForce(0, 256, make_float2(0.0f, 0.0f), densityColor, width, height);
		//addForce(200, 375, make_float2(0.0f, 0.0f), densityColor, width, height);
		//addForce(0, 460, make_float2(0.0f, 0.0f), densityColor, width, height);
	}
	visualizationStep(visualizationMethod, width, height);

	float2 p1 = make_float2((100 - width / 2.0) / (width / 2.0), (300 - height / 2.0) / (height / 2.0));
	float2 p2 = make_float2((201 - width / 2.0) / (width / 2.0), (401 - height / 2.0) / (height / 2.0));

	glColor3f(0.0f, 1.0f, 0.6f);
	//glRectf(p1.x, p1.y, p2.x, p2.y);

	glEnable(GL_DEPTH_TEST);
	glutSwapBuffers();
}

void idle() {
	glutPostRedisplay();
}

void keyDown(unsigned char key, int x, int y) {
	keysPressed[key] = true;
}

//Handling the user inputs and controlling the simulation
void keyUp(unsigned char key, int x, int y) {
	keysPressed[key] = false;
	switch (key) {
	case 'r':
		resetSimulationHost();
		break;

	case 'd':
		visualizationMethod = 0;
		break;
	case 'v':
		visualizationMethod = 1;
		break;
	case 'p':
		visualizationMethod = 2;
		break;

	case '1':
		densityColor.x = densityColor.y = densityColor.z = densityColor.w = 1.0f;
		break;

	case '2':
		densityColor.x = 1.0f;
		densityColor.y = densityColor.z = densityColor.w = 0.0f;
		break;

	case '3':
		densityColor.y = 1.0f;
		densityColor.x = densityColor.z = densityColor.w = 0.0f;
		break;

	case '4':
		densityColor.z = 1.0f;
		densityColor.x = densityColor.y = densityColor.w = 0.0f;
		break;

	case 27:
		exit(0);
		break;
	}
}

int mX, mY;

void mouseClick(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON)
		if (state == GLUT_DOWN) {
			mX = x;
			mY = y;
		}
}

//Handling the user input and controlling the simulation
void mouseMove(int x, int y) {
	force.x = 2 * (float)(x - mX);
	force.y = -2 * (float)(y - mY);
	//addForce(mX, height - mY, force, densityColor,width,height);
	addForce(256, 256, force * 2, densityColor, width, height);
	mX = x;
	mY = y;
}

//Handling the user input and controlling the simulation
void reshape(int newWidth, int newHeight) {
	width = newWidth;
	height = newHeight;
	glViewport(0, 0, width, height);
}

//Starting glut, openGL, simulation
int main(int argc, char* argv[]) {


	glutInit(&argc, argv);
	glutInitContextVersion(3, 0);
	glutInitContextFlags(GLUT_CORE_PROFILE | GLUT_DEBUG);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("GPGPU 13. labor: Incompressible fluid simulation");

	initOpenGL();

	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyDown);
	glutKeyboardUpFunc(keyUp);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMove);

	////CUDA processing
	//Default visualization (density) color
	densityColor.x = densityColor.y = densityColor.z = densityColor.w = 1.0f;
	//Init simulation
	initSimulation(width, height);

	glutMainLoop();
	return(0);
}