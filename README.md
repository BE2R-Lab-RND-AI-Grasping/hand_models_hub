# Models hub

The antropomorphic hand model for the project.

## Environment configuration
Dependencies are installed by ``` environment.yml ```

```bash
conda env create -f environment.yml
conda activate DIP-Flex_env
```

## Models

## 12hand
The project includes 12 DOF 3 fingered hand with fully opened kinematics

the project is at 

```bash
+--12hand
    |+-- Code
```

The hand model generation is used via [mjSpec](https://mujoco.readthedocs.io/en/stable/programming/modeledit.html)

run ```python run_hand.py```


## DIP-FLEX
The project has 1 model for closed-chained underactuatied fingers mechanism and 4 models with open-chained models. 2 scripts are added for the quick start.

```bash
+-- DIP-FLEX
    |+-- closed_chain
        |+-- assets
        |+-- DIP-Flex_closed_kinematics.xml # Underactuated five-bar spring-loaded mechanism
        |+-- run closed chain.ipynb # Quick start script
    |+-- opened_chain
        |+-- assets
        |+-- DIP-Flex.urdf # Oginal URDF model
        |+-- DIP-Flex_opened_kinematics.xml # Model without underactuations
        |+-- DIP-Flex_opened_kinematics_primitive.xml **(WIP)** # Model for faster collision simulation
        |+-- DIP-Flex_opened_kinematics_primitive_and_mesh.xml **(WIP)** # Model for faster collision simulation + original meshes for comparison
        |+-- run opened chain.ipynb # Quick start script
``` 


The **closed-chain** kinematics is the primary hand model ("DIP-Flex_closed_kinematics.xml" at left), utilizing an **underactuated five-bar spring-loaded mechanism**.

The **open-chain** kinematics model ("DIP-Flex_opened_kinematics.xml" at right) is provided to **enable control without underactuation**.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/4267919f-f0b3-4eed-8811-b2fb24fa1e07" height="300px" />
  <img src="https://github.com/user-attachments/assets/388905f2-74cc-44e9-936a-550df6bad1c4" height="300px" />
</p>

The **open-chain kinematics** model ("DIP-Flex_opened_kinematics_primitive.xml" at left), which replaces meshes **with primitives**, allows for **faster hand collision simulation**. 

For comparison, a model containing **both primitives and meshes** ("DIP-Flex_opened_kinematics_primitive_and_mesh.xml" at right) is also provided.   

<p align="center">
    <img src="https://github.com/user-attachments/assets/62238895-d168-49c5-8cf8-e7477b078d3d" height="350px" />
    <img src="https://github.com/user-attachments/assets/c1374c06-0b0e-4f89-b4a5-7d447af38d54" height="350px" />
</p>

