"""
Mapeamentos e constantes comuns usados em todo o projeto.
Inclui nomes de atividades, dispositivos e índices de colunas.
"""

# Mapeamento de IDs de atividades para nomes descritivos
ACTIVITY_NAMES = {
    1: "Stand",
    2: "Sit",
    3: "Sit and Talk",
    4: "Walk",
    5: "Walk and Talk",
    6: "Climb Stair",
    7: "Climb Stair and Talk",
    8: "Stand->Sit",
    9: "Sit->Stand",
    10: "Stand->Sit and Talk",
    11: "Sit->Stand and Talk",
    12: "Stand->Walk",
    13: "Walk->Stand",
    14: "Stand->Climb Stair",
    15: "Climb Stair->Walk",
    16: "Climb Stair and Talk->Walk and Talk"
}

# Mapeamento de IDs de dispositivos para nomes descritivos
DEVICE_NAMES = {
    1: "Pulso Esquerdo",
    2: "Pulso Direito",
    3: "Peito",
    4: "Perna Superior Direita",
    5: "Perna Inferior Esquerda"
}

# Índices das colunas no dataset
COL_DEVICE_ID = 0
COL_ACC_X = 1
COL_ACC_Y = 2
COL_ACC_Z = 3
COL_GYRO_X = 4
COL_GYRO_Y = 5
COL_GYRO_Z = 6
COL_MAG_X = 7
COL_MAG_Y = 8
COL_MAG_Z = 9
COL_TIMESTAMP = 10
COL_ACTIVITY = 11
