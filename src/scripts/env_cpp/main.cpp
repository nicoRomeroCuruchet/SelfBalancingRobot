#include "env.h"
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <ros/ros.h>
#include <sys/socket.h>
#include <netinet/in.h>


// comparar con PPO_52, esta version tenia un bug fuerte

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SelfBalancingRobotEnv");
    ros::NodeHandle nh;
    ros::Rate r(50);
    SelfBalancingRobotEnv env(nh, r);

     // Create a socket
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    // Bind the socket to an address and port
    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(12345); // Port number
    serverAddress.sin_addr.s_addr = INADDR_ANY;

    if (bind(serverSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) == -1) 
    {
        std::cerr << "Error binding socket" << std::endl;
        return 1;
    }

    // Listen for incoming connections
    if (listen(serverSocket, 1) == -1) 
    {
        std::cerr << "Error listening for connections" << std::endl;
        return 1;
    }

    std::cout << "Server is listening for incoming connections..." << std::endl;

    int clientSocket = 0;
    int count = 0;

    // Accept a connection
    clientSocket = accept(serverSocket, NULL, NULL);
    if (clientSocket == -1) {
        std::cerr << "Error accepting connection" << std::endl;
        return 1;
    }

    // Receive the serialized float array
    char buffer[1024];
    while (ros::ok()){

        ssize_t bytes_received = recv(clientSocket, buffer, sizeof(buffer), 0);
        int num_floats = bytes_received / sizeof(float);
        
        if (num_floats == 1) 
        {
            // Deserialize the received data into a vector of floats
            std::vector<float> received_values(num_floats);
            std::memcpy(received_values.data(), buffer, bytes_received);
            //std::cout << "action received: " << received_values[0] << "\n";
            std::shared_ptr<std::vector<float>> obs = env.step(received_values);
            float dataToSend[6] = {(*obs)[0], 
                                   (*obs)[1], 
                                   (*obs)[2], 
                                   (*obs)[3], 
                                   (*obs)[4], 
                                   (*obs)[5]}; 
            send(clientSocket, dataToSend, sizeof(dataToSend), 0);
            /*for(int i=0; i<6; i++){
                std::cout << " observation sent: " << std::fixed << std::setprecision(12)  << dataToSend[i] << "\n"; 
            }*/

        }
        else
        {            
            close(clientSocket);
            clientSocket = accept(serverSocket, NULL, NULL);
            std::cout<< "closing connection " << bytes_received << "\n";
        }
   
    }
}