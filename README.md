# Models hub

The antropomorphic hand model for the project

## Environment configuration
Dependencies are installed by ``` environment.yml ```

```bash
conda env create -f environment.yml
conda activate DIP-Flex_env
```

## Models

The project has 1 model for closed-chained underactuatied fingers mechanism and 4 models with open-chained models. 2 scripts are added for the quick start.

```bash
    |-- DIP-FLEX_code
    |-- assets

    |-- DIP-Flex_closed_kinematics.xml

    |-- DIP-Flex_opened_kinematics.urdf
    |-- DIP-Flex_opened_kinematics.xml
    |-- DIP-Flex_opened_kinematics_primitive.xml
    |-- DIP-Flex_opened_kinematics_primitive_and_mesh.xml

    |-- DIP-Flex run closed chain.ipynb
    |-- DIP-Flex run opened chain.ipynb
``` 



The closed-chain kinematics is the primary hand model ("DP-Flex_closed_kinematics.xml"), utilizing an underactuated five-bar spring-loaded mechanism.
The open-chain kinematics model ("DP-Flex_opened_kinematics.xml") is provided to enable control without underactuation.  

<img![image](https://github.com/user-attachments/assets/4267919f-f0b3-4eed-8811-b2fb24fa1e07) width="45%"/> ![image](https://github.com/user-attachments/assets/388905f2-74cc-44e9-936a-550df6bad1c4)


The open-chain kinematics model ("DP-Flex_opened_kinematics_primitive.xml"), which replaces meshes with primitives, allows for faster hand simulation. 

![image](https://github.com/user-attachments/assets/62238895-d168-49c5-8cf8-e7477b078d3d)


For comparison, a model containing both primitives and meshes ("DP-Flex_opened_kinematics_primitive_and_mesh.xml") is also provided.   

![image](https://github.com/user-attachments/assets/c1374c06-0b0e-4f89-b4a5-7d447af38d54)
