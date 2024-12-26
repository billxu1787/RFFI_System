# 项目介绍
这是一个开集RFFI系统的实现，包含被测设备的代码、接收机的代码和系统实现。

    
# 文件结构

    ├─README.md            
    ├─receiver
    │   ├─dummy_sdr_sig.m
    │   ├─LoRaPHY.m
    │   ├─lora_sdr_receiver.m
    │   ├─sdr_initial.m
    │   └─simulation.m   
    ├─system
    │   ├─client.py
    │   ├─dataset_utils.py
    │   ├─main_test.py
    │   ├─main_train.py
    │   ├─main_use.py
    │   ├─merge.py
    │   ├─models.py
    │   ├─temp.pth
    │   └─utils.py   
    └─transmitter
        ├─platformio.ini 
        └─examples
            └─Factory
                ├─  Factory.ino
                └─  utilities.h
                            
 
# 介绍
在安装相应的依赖项之后，运行main_train.py文件来训练特征提取器，或者直接使用预训练的特征提取器temp.pth。

运行main_test.py文件，从三个函数中选择一个：device_identify、device_disconnect和device_access。
device_identify函数可以根据现有支持集中的设备信息对查询集中的设备数据进行分类。
device_disconnect函数需要提前确定要断开连接的设备的标签，可以删除该标签对应的支持集中的所有数据，并对查询集中的设备数据进行分类。
device_access函数需要提供支持集、查询集和包含待接入设备数据的附加支持集，并可以通过将待接入设备的数据添加到支持集中来对查询集中设备的数据进行分类。

运行main_use.py文件，从三个函数中选择一个：device_identify、device_disconnect和device_access。
device_identify函数可以根据现有支持集中的设备信息对查询集中的设备数据进行分类。
device_disconnect函数需要提前确定待断开连接的设备的标签，其结果是删除该标签对应的支持集中的所有数据。
device_access函数需要提供支持集、查询集和包含待接入设备数据的附加支持集，其结果是将待接入设备的数据添加到支持集中。
 
# 代码来源
我们使用的发射器和接收器的代码分别是根据以下两个网站提供的代码进行修改的。

[发射机代码来源](https://www.ebyte.com/pdf-down/3298.html) 

[接收机代码来源](https://github.com/jkadbear/LoRaPHY)
 
 
 
