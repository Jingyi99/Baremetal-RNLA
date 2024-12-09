#define M_DIM 20
          #define K_DIM 1
          #define N_DIM 6
          
          typedef double data_t;
          
static data_t a_matrix [M_DIM*N_DIM] = {
  0.9125242884150524, 0.2618526048108707, 0.2522309100703436, 0.8227398627336125, 0.7566466389375209, 0.6164772665377012, 0.13463747123633585, 0.7354967634292587, 0.5915891614306871, 0.9578767525311169,
  0.11452831110610484, 0.5738647650751649, 0.08071819723851947, 0.04222830679534306, 0.6545516085569919, 0.6916324277590843, 0.5424758501357161, 0.1813777041902388, 0.9984198302177123, 0.021801973969179156,
  0.4636075442669806, 0.8259032534574547, 0.8221278474023749, 0.28922053856469154, 0.04582664640863787, 0.22125015960213557, 0.9660188422246867, 0.9770186878286297, 0.538190736610081, 0.8202618030853698,
  0.19309957782918408, 0.27098613189921605, 0.4080521892167669, 0.5656025557127724, 0.8340960960968165, 0.7906774426803732, 0.26293706175339815, 0.7991134497015253, 0.10046406941399999, 0.019449039876698326,
  0.007921818726771557, 0.1941769437030385, 0.71062783555477, 0.43263914937961023, 0.5693477087275669, 0.3204242298581954, 0.5596015134361886, 0.7675117828614614, 0.729019399346412, 0.8928129816035499,
  0.3852692170624098, 0.7225416266857754, 0.6302801575937815, 0.8367077392029664, 0.08425986431659793, 0.3894308373392179, 0.23737484289130673, 0.7205959105695015, 0.4824166658743615, 0.24337984434811588,
  0.05240818617556864, 0.7903124392165799, 0.46338182013337825, 0.9137599393284074, 0.1847908295813302, 0.27917226496834147, 0.6493254498005768, 0.4218543107870626, 0.48265117662731283, 0.19039607475207654,
  0.16575713726478036, 0.10089446130498658, 0.4901456961538695, 0.8963051899789372, 0.8697652014806521, 0.0834682077474952, 0.36653643754911014, 0.471106986105138, 0.9611894018126785, 0.8544199980547883,
  0.8813112351888436, 0.5913865256973991, 0.9767552525408504, 0.4091625943032051, 0.14211700098085545, 0.2736814942161243, 0.43631212084839, 0.4398953381186531, 0.9263980030130923, 0.8882554965580646,
  0.961901654791125, 0.0808232807661785, 0.11011279469619395, 0.10618325866542822, 0.08197180003610838, 0.07652692289735141, 0.7918299352164084, 0.023151840703882764, 0.5157937950495667, 0.7824130264629826,
  0.09591940900923723, 0.37506181974781017, 0.5777334351927227, 0.2474217062129761, 0.3144610310483029, 0.6414251819496957, 0.8538203235258695, 0.11184236758846056, 0.295504476313584, 0.8172498473756301,
  0.8622385446226385, 0.0807257704724712, 0.0895771145043931, 0.29306066482631266, 0.15840851210413343, 0.5564131185841314, 0.22323389971038576, 0.5532341909629258, 0.45956131120340926, 0.6838654325888245,
};
static data_t b_vec [20] = {
  0.2208898240369339, 0.1295845401922895, 0.7646326926059885, 0.8639736399812803, 0.9575839556930251, 0.793902336544066, 0.8924658348409262, 0.806220529518493, 0.5110297660342485, 0.13494266768012642,
  0.5528135992485499, 0.8358506120227736, 0.046011760866081386, 0.7193393324413144, 0.14023427944550304, 0.6793970630425875, 0.33385312438019665, 0.8789876178661603, 0.9592430554861505, 0.7916590666507345,
};
static data_t x_vec [6] = {
  0.2822535770068852, 0.18046687551818708, 0.4642390007858898, 0.11982706552681782, 0.20827431798067397, -0.12127189260522901,
};
