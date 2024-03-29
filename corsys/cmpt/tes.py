# -*- coding: utf-8 -*-
"""
    corsys.cmpt.tes
    ~~~~~~~~~~~~~~~


"""
from ..configs import Configurations
from . import Component


class ThermalEnergyStorage(Component):

    TYPE = 'tes'

    TEMPERATURE = 'tes_temp'
    TEMPERATURE_HEATING = 'tes_ht_temp'
    TEMPERATURE_DOMESTIC = 'tes_dom_temp'

    # noinspection PyProtectedMember
    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        self.volume = configs.getfloat('General', 'volume')

        # For the thermal storage capacity in kWh/K, it will be assumed to be filled with water,
        # resulting in a specific heat capacity of 4.184 J/g*K.
        # TODO: Make tank content and specific heat capacity configurable
        self.capacity = 4.184*self.volume/3600

    @property
    def type(self) -> str:
        return self.TYPE
