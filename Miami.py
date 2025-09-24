# miami_florida_real_estate.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MiamiRealEstateAnalyzer:
    def __init__(self, area_name):
        self.area = area_name
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', 
                      '#AB83A1', '#8B0000', '#228B22', '#FFD700', '#0038A8']
        
        self.start_year = 2002
        self.end_year = 2025
        
        # Configuration spécifique à chaque zone de Miami/Floride
        self.config = self._get_area_config()
        
    def _get_area_config(self):
        """Retourne la configuration spécifique pour chaque zone de Miami/Floride"""
        configs = {
            "Miami Downtown": {
                "population_base": 450000,
                "budget_base": 2800,
                "type": "urban_core",
                "specialites": ["finance", "tourisme", "croisières", "luxe", "condos"],
                "prix_m2_base": 8500,
                "segment_immobilier": "luxury_condo",
                "currency": "USD",
                "key_features": ["skyline", "waterfront", "international_buyers"]
            },
            "Miami Beach": {
                "population_base": 92000,
                "budget_base": 1800,
                "type": "coastal_luxury",
                "specialites": ["tourisme", "plage", "luxe", "divertissement", "art_deco"],
                "prix_m2_base": 11000,
                "segment_immobilier": "premium_beachfront",
                "currency": "USD",
                "key_features": ["beachfront", "nightlife", "art_deco_architecture"]
            },
            "Brickell": {
                "population_base": 35000,
                "budget_base": 2200,
                "type": "financial_district",
                "specialites": ["finance", "affaires", "condos_luxe", "international", "banques"],
                "prix_m2_base": 9500,
                "segment_immobilier": "financial_luxury",
                "currency": "USD",
                "key_features": ["financial_center", "high_rises", "young_professionals"]
            },
            "Coral Gables": {
                "population_base": 50000,
                "budget_base": 1500,
                "type": "upscale_residential",
                "specialites": ["residentiel_haut_gamme", "education", "architecture", "arbres", "calme"],
                "prix_m2_base": 7500,
                "segment_immobilier": "premium_suburban",
                "currency": "USD",
                "key_features": ["historic", "tree_canopy", "upscale_residential"]
            },
            "Fort Lauderdale": {
                "population_base": 180000,
                "budget_base": 2200,
                "type": "coastal_marine",
                "specialites": ["marina", "tourisme", "plage", "yachting", "residentiel"],
                "prix_m2_base": 6000,
                "segment_immobilier": "marine_lifestyle",
                "currency": "USD",
                "key_features": ["yachting_capital", "beaches", "waterways"]
            },
            "West Palm Beach": {
                "population_base": 110000,
                "budget_base": 1900,
                "type": "affluent_suburban",
                "specialites": ["retraite", "luxe", "golf", "culture", "residentiel"],
                "prix_m2_base": 5500,
                "segment_immobilier": "affluent_retirement",
                "currency": "USD",
                "key_features": ["golf_communities", "cultural_venues", "affluent_retirees"]
            },
            "South Florida Region": {
                "population_base": 6000000,
                "budget_base": 8500,
                "type": "tropical_metropolitan",
                "specialites": ["tourisme", "retraite", "international", "agriculture_tropicale", "sante"],
                "prix_m2_base": 5000,
                "segment_immobilier": "mixed_tropical",
                "currency": "USD",
                "key_features": ["tropical_climate", "international_hub", "retirement_destination"]
            },
            # Configuration par défaut
            "default": {
                "population_base": 100000,
                "budget_base": 1200,
                "type": "florida_coastal",
                "specialites": ["tourisme", "residentiel", "services"],
                "prix_m2_base": 4500,
                "segment_immobilier": "coastal_mixed",
                "currency": "USD",
                "key_features": ["beach_access", "tourist_destination"]
            }
        }
        
        return configs.get(self.area, configs["default"])
    
    def generate_financial_data(self):
        """Génère des données financières et immobilières pour la zone de Miami/Floride"""
        print(f"🌴 Génération des données financières et immobilières pour {self.area}, Floride...")
        
        # Créer une base de données annuelle
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='Y')
        
        data = {'Year': [date.year for date in dates]}
        
        # Données démographiques
        data['Population'] = self._simulate_population(dates)
        data['Households'] = self._simulate_households(dates)
        data['Median_Income'] = self._simulate_median_income(dates)
        data['International_Buyers_Percentage'] = self._simulate_international_buyers(dates)
        
        # Recettes municipales (en millions de dollars)
        data['Total_Revenue'] = self._simulate_total_revenue(dates)
        data['Property_Tax_Revenue'] = self._simulate_property_tax_revenue(dates)
        data['Tourism_Tax_Revenue'] = self._simulate_tourism_tax_revenue(dates)
        data['Sales_Tax_Revenue'] = self._simulate_sales_tax_revenue(dates)
        data['Other_Revenue'] = self._simulate_other_revenue(dates)
        
        # Dépenses municipales
        data['Total_Expenses'] = self._simulate_total_expenses(dates)
        data['Infrastructure_Expenses'] = self._simulate_infrastructure_expenses(dates)
        data['Public_Safety_Expenses'] = self._simulate_public_safety_expenses(dates)
        data['Beach_Maintenance_Expenses'] = self._simulate_beach_maintenance_expenses(dates)
        data['Climate_Resilience_Expenses'] = self._simulate_climate_resilience_expenses(dates)
        
        # Indicateurs financiers
        data['Budget_Balance'] = self._simulate_budget_balance(dates)
        data['Municipal_Debt'] = self._simulate_municipal_debt(dates)
        data['Debt_Ratio'] = self._simulate_debt_ratio(dates)
        
        # Données immobilières (spécifiques à Miami/Floride)
        data['Median_Home_Price'] = self._simulate_median_home_price(dates)
        data['Price_per_Sqft'] = self._simulate_price_per_sqft(dates)
        data['Condo_Price_per_Sqft'] = self._simulate_condo_prices(dates)
        data['Home_Sales_Volume'] = self._simulate_home_sales(dates)
        data['New_Construction_Permits'] = self._simulate_construction_permits(dates)
        data['Rental_Vacancy_Rate'] = self._simulate_vacancy_rate(dates)
        data['Average_Rent'] = self._simulate_average_rent(dates)
        data['Beachfront_Premium'] = self._simulate_beachfront_premium(dates)
        
        # Investissements spécifiques adaptés à Miami/Floride
        data['Real_Estate_Development'] = self._simulate_real_estate_development(dates)
        data['Tourism_Infrastructure_Investment'] = self._simulate_tourism_infrastructure_investment(dates)
        data['Climate_Adaptation_Investment'] = self._simulate_climate_adaptation_investment(dates)
        data['Luxury_Development_Investment'] = self._simulate_luxury_development_investment(dates)
        data['Marina_Waterfront_Investment'] = self._simulate_marina_investment(dates)
        
        df = pd.DataFrame(data)
        
        # Ajouter des tendances spécifiques au marché floridien
        self._add_florida_trends(df)
        
        return df
    
    def _simulate_population(self, dates):
        """Simule la population de la zone"""
        base_population = self.config["population_base"]
        
        population = []
        for i, date in enumerate(dates):
            # Croissance démographique floridienne (forte attractivité)
            if self.config["type"] in ["urban_core", "coastal_luxury"]:
                growth_rate = 0.018  # Forte croissance à Miami
            elif self.config["type"] == "financial_district":
                growth_rate = 0.022  # Très forte croissance à Brickell
            else:
                growth_rate = 0.015  # Croissance floridienne typique
                
            growth = 1 + growth_rate * i
            population.append(base_population * growth)
        
        return population
    
    def _simulate_households(self, dates):
        """Simule le nombre de ménages"""
        base_households = self.config["population_base"] / 2.2  # Taille moyenne des ménages en Floride
        
        households = []
        for i, date in enumerate(dates):
            growth = 1 + 0.016 * i
            households.append(base_households * growth)
        
        return households
    
    def _simulate_median_income(self, dates):
        """Simule le revenu médian"""
        # Revenu médian de base selon la zone
        if self.config["type"] in ["financial_district", "coastal_luxury"]:
            base_income = 85000
        elif self.config["type"] == "urban_core":
            base_income = 65000
        else:
            base_income = 55000
        
        incomes = []
        for i, date in enumerate(dates):
            year = date.year
            # Croissance du revenu avec des variations spécifiques à la Floride
            if 2002 <= year <= 2007:
                growth = 1 + 0.035 * (year - 2002)
            elif 2008 <= year <= 2011:
                growth = 1 - 0.04 * (year - 2008)  # Forte baisse pendant la crise
            elif 2012 <= year <= 2019:
                growth = 1 + 0.04 * (year - 2012)
            elif 2020 <= year <= 2021:
                growth = 1 - 0.01  # Impact modéré du COVID
            else:
                growth = 1 + 0.045 * (year - 2022)  # Forte reprise post-COVID
            
            noise = np.random.normal(1, 0.06)
            incomes.append(base_income * growth * noise)
        
        return incomes
    
    def _simulate_international_buyers(self, dates):
        """Simule le pourcentage d'acheteurs internationaux (spécifique à Miami)"""
        percentages = []
        for i, date in enumerate(dates):
            year = date.year
            base_percentage = 25.0  # Base de 25% d'acheteurs internationaux
            
            # Variations selon les conditions économiques globales
            if 2002 <= year <= 2007:
                # Période de forte demande internationale
                multiplier = 1 + 0.05 * (year - 2002)
            elif 2008 <= year <= 2009:
                # Crise réduit les achats internationaux
                multiplier = 0.70
            elif 2010 <= year <= 2014:
                # Reprise progressive
                multiplier = 1 + 0.04 * (year - 2010)
            elif 2015 <= year <= 2019:
                # Forte demande d'Amérique latine et d'Europe
                multiplier = 1.3
            elif 2020 <= year <= 2021:
                # COVID réduit les voyages internationaux
                multiplier = 0.60
            else:
                # Retour fort des acheteurs internationaux
                multiplier = 1.4
            
            noise = np.random.normal(1, 0.08)
            percentage = base_percentage * multiplier * noise
            percentages.append(min(60.0, max(10.0, percentage)))  # Entre 10% et 60%
        
        return percentages
    
    def _simulate_total_revenue(self, dates):
        """Simule les recettes totales"""
        base_revenue = self.config["budget_base"]
        
        revenue = []
        for i, date in enumerate(dates):
            # Croissance économique floridienne
            if self.config["type"] in ["urban_core", "financial_district"]:
                growth_rate = 0.050  # Croissance très forte à Miami
            else:
                growth_rate = 0.042  # Croissance forte en Floride
                
            growth = 1 + growth_rate * i
            noise = np.random.normal(1, 0.09)
            revenue.append(base_revenue * growth * noise)
        
        return revenue
    
    def _simulate_property_tax_revenue(self, dates):
        """Simule les recettes de taxe foncière"""
        base_tax = self.config["budget_base"] * 0.40
        
        tax_revenue = []
        for i, date in enumerate(dates):
            growth = 1 + 0.038 * i
            noise = np.random.normal(1, 0.07)
            tax_revenue.append(base_tax * growth * noise)
        
        return tax_revenue
    
    def _simulate_tourism_tax_revenue(self, dates):
        """Simule les recettes de taxe touristique (spécifique à la Floride)"""
        base_tourism_tax = self.config["budget_base"] * 0.25
        
        multiplier = 2.0 if "tourisme" in self.config["specialites"] else 0.8
        
        tourism_tax = []
        for i, date in enumerate(dates):
            year = date.year
            # Variations saisonnières et cycliques
            if year in [2005, 2010, 2015, 2020]:
                year_multiplier = 0.9  # Années de ralentissement
            elif year in [2007, 2012, 2017, 2022]:
                year_multiplier = 1.3  # Années de forte fréquentation
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.045 * i
            noise = np.random.normal(1, 0.15)
            tourism_tax.append(base_tourism_tax * growth * year_multiplier * multiplier * noise)
        
        return tourism_tax
    
    def _simulate_sales_tax_revenue(self, dates):
        """Simule les recettes de taxe de vente"""
        base_sales_tax = self.config["budget_base"] * 0.20
        
        sales_tax = []
        for i, date in enumerate(dates):
            growth = 1 + 0.040 * i
            noise = np.random.normal(1, 0.08)
            sales_tax.append(base_sales_tax * growth * noise)
        
        return sales_tax
    
    def _simulate_other_revenue(self, dates):
        """Simule les autres recettes"""
        base_other = self.config["budget_base"] * 0.15
        
        other_revenue = []
        for i, date in enumerate(dates):
            growth = 1 + 0.035 * i
            noise = np.random.normal(1, 0.10)
            other_revenue.append(base_other * growth * noise)
        
        return other_revenue
    
    def _simulate_total_expenses(self, dates):
        """Simule les dépenses totales"""
        base_expenses = self.config["budget_base"] * 0.95
        
        expenses = []
        for i, date in enumerate(dates):
            growth = 1 + 0.044 * i
            noise = np.random.normal(1, 0.08)
            expenses.append(base_expenses * growth * noise)
        
        return expenses
    
    def _simulate_infrastructure_expenses(self, dates):
        """Simule les dépenses d'infrastructure"""
        base_infra = self.config["budget_base"] * 0.30
        
        infra_expenses = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2005, 2012, 2018, 2023]:
                multiplier = 1.7  # Années de gros travaux
            else:
                multiplier = 1.0
            
            growth = 1 + 0.042 * i
            noise = np.random.normal(1, 0.16)
            infra_expenses.append(base_infra * growth * multiplier * noise)
        
        return infra_expenses
    
    def _simulate_public_safety_expenses(self, dates):
        """Simule les dépenses de sécurité publique"""
        base_safety = self.config["budget_base"] * 0.25
        
        safety_expenses = []
        for i, date in enumerate(dates):
            growth = 1 + 0.038 * i
            noise = np.random.normal(1, 0.06)
            safety_expenses.append(base_safety * growth * noise)
        
        return safety_expenses
    
    def _simulate_beach_maintenance_expenses(self, dates):
        """Simule les dépenses d'entretien des plages (spécifique à la Floride)"""
        base_beach = self.config["budget_base"] * 0.08
        
        multiplier = 1.8 if "plage" in self.config["specialites"] else 0.5
        
        beach_expenses = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2004, 2010, 2016, 2022]:
                year_multiplier = 2.2  # Renourissement des plages
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.040 * i
            noise = np.random.normal(1, 0.20)
            beach_expenses.append(base_beach * growth * year_multiplier * multiplier * noise)
        
        return beach_expenses
    
    def _simulate_climate_resilience_expenses(self, dates):
        """Simule les dépenses de résilience climatique (spécifique à la Floride)"""
        base_climate = self.config["budget_base"] * 0.12
        
        climate_expenses = []
        for i, date in enumerate(dates):
            year = date.year
            if year >= 2015:
                # Augmentation significative après les ouragans majeurs
                acceleration = 1 + 0.10 * (year - 2015)
            else:
                acceleration = 1.0
            
            growth = 1 + 0.050 * i
            noise = np.random.normal(1, 0.18)
            climate_expenses.append(base_climate * growth * acceleration * noise)
        
        return climate_expenses
    
    def _simulate_budget_balance(self, dates):
        """Simule l'équilibre budgétaire"""
        balance = []
        for i, date in enumerate(dates):
            base_balance = self.config["budget_base"] * 0.05
            
            year = date.year
            if year >= 2010:
                improvement = 1 + 0.015 * (year - 2010)
            else:
                improvement = 1
            
            noise = np.random.normal(1, 0.22)
            balance.append(base_balance * improvement * noise)
        
        return balance
    
    def _simulate_municipal_debt(self, dates):
        """Simule la dette municipale"""
        base_debt = self.config["budget_base"] * 0.65
        
        debt = []
        for i, date in enumerate(dates):
            year = date.year
            if year >= 2012:
                reduction = 1 - 0.012 * (year - 2012)
            else:
                reduction = 1.0
            
            noise = np.random.normal(1, 0.11)
            debt.append(base_debt * reduction * noise)
        
        return debt
    
    def _simulate_debt_ratio(self, dates):
        """Simule le ratio d'endettement"""
        ratios = []
        for i, date in enumerate(dates):
            base_ratio = 0.60
            
            year = date.year
            if year >= 2010:
                improvement = 1 - 0.016 * (year - 2010)
            else:
                improvement = 1
            
            noise = np.random.normal(1, 0.09)
            ratios.append(base_ratio * improvement * noise)
        
        return ratios
    
    def _simulate_median_home_price(self, dates):
        """Simule le prix médian des maisons (spécifique à Miami/Floride)"""
        base_price = self.config["prix_m2_base"] * 180  # Pour une maison moyenne de 180m²
        
        prices = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance du marché immobilier floridien
            if self.config["segment_immobilier"] == "luxury_condo":
                growth_rate = 0.068  # Croissance très forte pour le luxe
            elif self.config["segment_immobilier"] == "premium_beachfront":
                growth_rate = 0.072  # Croissance exceptionnelle en front de mer
            elif self.config["segment_immobilier"] == "financial_luxury":
                growth_rate = 0.065  # Croissance forte dans les quartiers financiers
            else:
                growth_rate = 0.055  # Croissance floridienne typique
            
            # Ajustements annuels basés sur des événements réels
            if 2002 <= year <= 2006:
                # Boom immobilier pré-crise (spéculation en Floride)
                multiplier = 1 + 0.15 * (year - 2002)
            elif 2007 <= year <= 2011:
                # Crise des subprimes - impact dévastateur en Floride
                multiplier = 0.60  # Effondrement des prix
            elif 2012 <= year <= 2019:
                # Reprise progressive avec afflux d'acheteurs internationaux
                multiplier = 1 + 0.12 * (year - 2012)
            elif 2020 <= year <= 2021:
                # COVID-19 - exode vers la Floride
                multiplier = 1.15  # Hausse exceptionnelle
            else:
                # Poursuite de la forte demande
                multiplier = 1 + 0.10 * (year - 2022)
            
            growth = 1 + growth_rate * i
            noise = np.random.normal(1, 0.16)
            prices.append(base_price * growth * multiplier * noise)
        
        return prices
    
    def _simulate_price_per_sqft(self, dates):
        """Simule le prix au pied carré"""
        base_price_sqft = self.config["prix_m2_base"] / 10.764  # Conversion m² → pieds carrés
        
        prices = []
        for i, date in enumerate(dates):
            # Suit la même tendance que le prix médian
            median_price = self._simulate_median_home_price([date])[0]
            avg_home_size = 180 * 10.764  # 180m² en pieds carrés
            prices.append(median_price / avg_home_size)
        
        return prices
    
    def _simulate_condo_prices(self, dates):
        """Simule les prix des condos (spécifique à Miami)"""
        base_condo_price = self.config["prix_m2_base"] / 10.764 * 1.2  # 20% de premium pour les condos
        
        prices = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Les condos ont une dynamique différente des maisons
            if 2002 <= year <= 2006:
                multiplier = 1 + 0.18 * (year - 2002)  # Boom des condos
            elif 2007 <= year <= 2011:
                multiplier = 0.55  # Effondrement plus marqué
            elif 2012 <= year <= 2019:
                multiplier = 1 + 0.14 * (year - 2012)  # Reprise plus forte
            elif 2020 <= year <= 2021:
                multiplier = 1.25  # Hausse très forte pendant le COVID
            else:
                multiplier = 1 + 0.11 * (year - 2022)
            
            growth = 1 + 0.070 * i
            noise = np.random.normal(1, 0.18)
            prices.append(base_condo_price * growth * multiplier * noise)
        
        return prices
    
    def _simulate_home_sales(self, dates):
        """Simule le volume des ventes immobilières"""
        base_sales = self.config["population_base"] / 100
        
        sales = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Variations selon la conjoncture floridienne
            if 2002 <= year <= 2005:
                multiplier = 1 + 0.15 * (year - 2002)
            elif 2006 <= year <= 2010:
                multiplier = 0.50  # Effondrement du marché
            elif 2011 <= year <= 2019:
                multiplier = 1 + 0.10 * (year - 2011)
            elif 2020 <= year <= 2021:
                multiplier = 1.20  # Forte activité pendant le COVID
            else:
                multiplier = 1 + 0.08 * (year - 2022)
            
            growth = 1 + 0.014 * i
            noise = np.random.normal(1, 0.20)
            sales.append(base_sales * growth * multiplier * noise)
        
        return sales
    
    def _simulate_construction_permits(self, dates):
        """Simule les permis de construction"""
        base_permits = self.config["population_base"] / 500
        
        permits = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year in [2005, 2013, 2018, 2021, 2024]:
                multiplier = 2.0  # Années de forte construction
            elif year in [2008, 2011, 2020]:
                multiplier = 0.4  # Années de ralentissement
            else:
                multiplier = 1.0
            
            growth = 1 + 0.020 * i
            noise = np.random.normal(1, 0.28)
            permits.append(base_permits * growth * multiplier * noise)
        
        return permits
    
    def _simulate_vacancy_rate(self, dates):
        """Simule le taux d'inoccupation locative"""
        base_vacancy = 6.0  # 6% de base en Floride (marché saisonnier)
        
        vacancies = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Variations selon le marché locatif floridien
            if 2002 <= year <= 2006:
                rate = base_vacancy - 0.8 * (year - 2002)  # Forte demande locative
            elif 2007 <= year <= 2011:
                rate = base_vacancy + 3.0  # Hausse importante pendant la crise
            elif 2012 <= year <= 2019:
                rate = base_vacancy - 0.4 * (year - 2012)  # Baisse progressive
            elif 2020 <= year <= 2021:
                rate = base_vacancy - 1.0  # Forte demande pendant le COVID
            else:
                rate = base_vacancy - 0.3 * (year - 2022)  # Demande soutenue
            
            noise = np.random.normal(0, 0.4)
            vacancies.append(max(2.0, rate + noise))  # Minimum 2%
        
        return vacancies
    
    def _simulate_average_rent(self, dates):
        """Simule le loyer moyen"""
        base_rent = self.config["prix_m2_base"] / 40  # Relation prix/loyer ajustée pour la Floride
        
        rents = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Suit la tendance des prix mais avec moins de volatilité
            if 2002 <= year <= 2007:
                growth = 1 + 0.06 * (year - 2002)
            elif 2008 <= year <= 2010:
                growth = 1 - 0.03 * (year - 2008)  # Baisse modérée
            elif 2011 <= year <= 2019:
                growth = 1 + 0.05 * (year - 2011)
            elif 2020 <= year <= 2021:
                growth = 1.08  # Forte hausse pendant le COVID
            else:
                growth = 1 + 0.06 * (year - 2022)
            
            noise = np.random.normal(1, 0.09)
            rents.append(base_rent * growth * noise)
        
        return rents
    
    def _simulate_beachfront_premium(self, dates):
        """Simule le premium pour les propriétés en front de mer"""
        premiums = []
        for i, date in enumerate(dates):
            year = date.year
            base_premium = 50.0  # 50% de premium de base
            
            # Le premium fluctue selon la demande
            if 2002 <= year <= 2006:
                premium = base_premium + 5 * (year - 2002)
            elif 2007 <= year <= 2011:
                premium = base_premium - 10  # Réduction pendant la crise
            elif 2012 <= year <= 2019:
                premium = base_premium + 8 * (year - 2012)
            elif 2020 <= year <= 2021:
                premium = base_premium + 15  # Forte hausse pendant le COVID
            else:
                premium = base_premium + 20  # Demande soutenue
            
            noise = np.random.normal(0, 3)
            premiums.append(max(30.0, premium + noise))  # Minimum 30%
        
        return premiums
    
    def _simulate_real_estate_development(self, dates):
        """Simule l'investissement dans le développement immobilier"""
        base_development = self.config["budget_base"] * 0.15
        
        multiplier = 1.8 if "condos" in self.config["specialites"] else 1.0
        
        development = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2005, 2013, 2018, 2022]:
                year_multiplier = 2.2
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.055 * i
            noise = np.random.normal(1, 0.22)
            development.append(base_development * growth * year_multiplier * multiplier * noise)
        
        return development
    
    def _simulate_tourism_infrastructure_investment(self, dates):
        """Simule l'investissement dans l'infrastructure touristique"""
        base_investment = self.config["budget_base"] * 0.12
        
        multiplier = 2.2 if "tourisme" in self.config["specialites"] else 0.7
        
        investment = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2006, 2012, 2018, 2024]:
                year_multiplier = 1.8
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.048 * i
            noise = np.random.normal(1, 0.19)
            investment.append(base_investment * growth * year_multiplier * multiplier * noise)
        
        return investment
    
    def _simulate_climate_adaptation_investment(self, dates):
        """Simule l'investissement dans l'adaptation climatique"""
        base_investment = self.config["budget_base"] * 0.10
        
        investment = []
        for i, date in enumerate(dates):
            year = date.year
            if year >= 2015:
                # Augmentation significative après les ouragans
                acceleration = 1 + 0.12 * (year - 2015)
            else:
                acceleration = 1.0
            
            growth = 1 + 0.060 * i
            noise = np.random.normal(1, 0.25)
            investment.append(base_investment * growth * acceleration * noise)
        
        return investment
    
    def _simulate_luxury_development_investment(self, dates):
        """Simule l'investissement dans le développement de luxe"""
        base_investment = self.config["budget_base"] * 0.08
        
        multiplier = 2.5 if "luxe" in self.config["specialites"] else 0.5
        
        investment = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2007, 2014, 2020]:
                year_multiplier = 1.9
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.065 * i
            noise = np.random.normal(1, 0.23)
            investment.append(base_investment * growth * year_multiplier * multiplier * noise)
        
        return investment
    
    def _simulate_marina_investment(self, dates):
        """Simule l'investissement dans les marinas et waterfront"""
        base_investment = self.config["budget_base"] * 0.06
        
        multiplier = 1.7 if "marina" in self.config["specialites"] else 0.6
        
        investment = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2008, 2015, 2021]:
                year_multiplier = 1.6
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.042 * i
            noise = np.random.normal(1, 0.20)
            investment.append(base_investment * growth * year_multiplier * multiplier * noise)
        
        return investment
    
    def _add_florida_trends(self, df):
        """Ajoute des tendances réalistes adaptées au marché floridien"""
        for i, row in df.iterrows():
            year = row['Year']
            
            # Crise des subprimes (2007-2011) - impact majeur en Floride
            if 2007 <= year <= 2011:
                df.loc[i, 'Median_Home_Price'] *= 0.60
                df.loc[i, 'Home_Sales_Volume'] *= 0.50
                df.loc[i, 'New_Construction_Permits'] *= 0.40
            
            # Ouragan Wilma (2005) et saison cyclonique active
            if year == 2005:
                df.loc[i, 'Climate_Resilience_Expenses'] *= 1.8
                df.loc[i, 'Beach_Maintenance_Expenses'] *= 2.0
            
            # Afflux d'acheteurs internationaux (2012-2019)
            if 2012 <= year <= 2019:
                df.loc[i, 'International_Buyers_Percentage'] *= 1.4
                df.loc[i, 'Luxury_Development_Investment'] *= 1.6
            
            # Ouragan Irma (2017)
            if year == 2017:
                df.loc[i, 'Climate_Adaptation_Investment'] *= 2.2
                df.loc[i, 'Insurance_Costs'] = df.loc[i, 'Median_Home_Price'] * 0.02  # Coût assurance
            
            # COVID-19 et exode vers la Floride (2020-2021)
            if 2020 <= year <= 2021:
                if year == 2020:
                    df.loc[i, 'Tourism_Tax_Revenue'] *= 0.50
                    df.loc[i, 'Home_Sales_Volume'] *= 1.20
                else:
                    df.loc[i, 'Median_Home_Price'] *= 1.15
                    df.loc[i, 'Average_Rent'] *= 1.08
                    df.loc[i, 'Population'] *= 1.03  # Afflux de nouveaux résidents
            
            # Pénurie d'assurance et hausse des primes (2022-2025)
            if year >= 2022:
                df.loc[i, 'Climate_Adaptation_Investment'] *= 1.3
                df.loc[i, 'Beachfront_Premium'] *= 0.95  # Léger ajustement due aux risques
            
            # Développement continu des croisières (Miami)
            if year >= 2010 and "croisières" in self.config["specialites"]:
                df.loc[i, 'Tourism_Infrastructure_Investment'] *= 1.4
                df.loc[i, 'Tourism_Tax_Revenue'] *= 1.3
    
    def create_financial_analysis(self, df):
        """Crée une analyse complète des finances et de l'immobilier miamien"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 28))
        
        # 1. Évolution des prix immobiliers
        ax1 = plt.subplot(5, 2, 1)
        self._plot_real_estate_prices(df, ax1)
        
        # 2. Marché des condos et front de mer
        ax2 = plt.subplot(5, 2, 2)
        self._plot_condo_beachfront_market(df, ax2)
        
        # 3. Activité immobilière
        ax3 = plt.subplot(5, 2, 3)
        self._plot_real_estate_activity(df, ax3)
        
        # 4. Recettes touristiques et internationales
        ax4 = plt.subplot(5, 2, 4)
        self._plot_tourism_international_revenue(df, ax4)
        
        # 5. Marché locatif
        ax5 = plt.subplot(5, 2, 5)
        self._plot_rental_market(df, ax5)
        
        # 6. Investissements spécifiques à Miami
        ax6 = plt.subplot(5, 2, 6)
        self._plot_miami_investments(df, ax6)
        
        # 7. Démographie et acheteurs internationaux
        ax7 = plt.subplot(5, 2, 7)
        self._plot_demography_international(df, ax7)
        
        # 8. Dépenses climatiques et résilience
        ax8 = plt.subplot(5, 2, 8)
        self._plot_climate_resilience(df, ax8)
        
        # 9. Construction et développement
        ax9 = plt.subplot(5, 2, 9)
        self._plot_construction_development(df, ax9)
        
        # 10. Investissements sectoriels
        ax10 = plt.subplot(5, 2, 10)
        self._plot_sectorial_investments(df, ax10)
        
        plt.suptitle(f'Financial and Real Estate Analysis of {self.area}, Florida ({self.start_year}-{self.end_year})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.area.replace(" ", "_").lower()}_florida_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Générer les insights
        self._generate_miami_insights(df)
    
    def _plot_real_estate_prices(self, df, ax):
        """Plot de l'évolution des prix immobiliers"""
        ax.plot(df['Year'], df['Median_Home_Price']/1000, label='Median Home Price', 
               linewidth=3, color='#FF6B6B', alpha=0.8)
        ax.plot(df['Year'], df['Price_per_Sqft'], label='Price per Sqft', 
               linewidth=2, color='#4ECDC4', alpha=0.8, linestyle='--')
        
        ax.set_title('Real Estate Price Evolution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (Thousand $) / ($ per Sqft)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ajouter des annotations pour les événements marquants
        ax.annotate('Subprime Crisis', xy=(2008, df.loc[df['Year'] == 2008, 'Median_Home_Price'].values[0]/1000), 
                   xytext=(2008, df.loc[df['Year'] == 2008, 'Median_Home_Price'].values[0]/1000 * 0.6),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.annotate('COVID Boom', xy=(2021, df.loc[df['Year'] == 2021, 'Median_Home_Price'].values[0]/1000), 
                   xytext=(2021, df.loc[df['Year'] == 2021, 'Median_Home_Price'].values[0]/1000 * 1.3),
                   arrowprops=dict(arrowstyle='->', color='green'))
    
    def _plot_condo_beachfront_market(self, df, ax):
        """Plot du marché des condos et front de mer"""
        ax.plot(df['Year'], df['Condo_Price_per_Sqft'], label='Condo Price per Sqft', 
               linewidth=2, color='#45B7D1', alpha=0.8)
        
        ax.set_title('Condo and Beachfront Market', fontsize=12, fontweight='bold')
        ax.set_ylabel('Condo Price ($/Sqft)', color='#45B7D1')
        ax.tick_params(axis='y', labelcolor='#45B7D1')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(df['Year'], df['Beachfront_Premium'], label='Beachfront Premium', 
                linewidth=2, color='#F9A602', alpha=0.8)
        ax2.set_ylabel('Beachfront Premium (%)', color='#F9A602')
        ax2.tick_params(axis='y', labelcolor='#F9A602')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_real_estate_activity(self, df, ax):
        """Plot de l'activité immobilière"""
        ax.bar(df['Year'], df['Home_Sales_Volume'], label='Home Sales', 
              color='#4ECDC4', alpha=0.7)
        
        ax.set_title('Real Estate Market Activity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Home Sales Volume', color='#4ECDC4')
        ax.tick_params(axis='y', labelcolor='#4ECDC4')
        ax.grid(True, alpha=0.3, axis='y')
        
        ax2 = ax.twinx()
        ax2.plot(df['Year'], df['New_Construction_Permits'], label='Construction Permits', 
                linewidth=2, color='#FF6B6B')
        ax2.set_ylabel('Construction Permits', color='#FF6B6B')
        ax2.tick_params(axis='y', labelcolor='#FF6B6B')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_tourism_international_revenue(self, df, ax):
        """Plot des recettes touristiques et internationales"""
        ax.plot(df['Year'], df['Tourism_Tax_Revenue'], label='Tourism Tax Revenue', 
               linewidth=2, color='#F9A602', alpha=0.8)
        
        ax.set_title('Tourism and International Revenue', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tourism Tax Revenue (M$)', color='#F9A602')
        ax.tick_params(axis='y', labelcolor='#F9A602')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(df['Year'], df['International_Buyers_Percentage'], label='International Buyers', 
                linewidth=2, color='#45B7D1', alpha=0.8)
        ax2.set_ylabel('International Buyers (%)', color='#45B7D1')
        ax2.tick_params(axis='y', labelcolor='#45B7D1')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_rental_market(self, df, ax):
        """Plot du marché locatif"""
        ax.plot(df['Year'], df['Average_Rent'], label='Average Rent', 
               linewidth=2, color='#6A0572', alpha=0.8)
        
        ax.set_title('Rental Market Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Rent ($)', color='#6A0572')
        ax.tick_params(axis='y', labelcolor='#6A0572')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(df['Year'], df['Rental_Vacancy_Rate'], label='Vacancy Rate', 
                linewidth=2, color='#FF6B6B', alpha=0.8)
        ax2.set_ylabel('Vacancy Rate (%)', color='#FF6B6B')
        ax2.tick_params(axis='y', labelcolor='#FF6B6B')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_miami_investments(self, df, ax):
        """Plot des investissements spécifiques à Miami"""
        ax.plot(df['Year'], df['Luxury_Development_Investment'], label='Luxury Development', 
               linewidth=2, color='#FF6B6B', alpha=0.8)
        ax.plot(df['Year'], df['Tourism_Infrastructure_Investment'], label='Tourism Infrastructure', 
               linewidth=2, color='#F9A602', alpha=0.8)
        ax.plot(df['Year'], df['Marina_Waterfront_Investment'], label='Marina/Waterfront', 
               linewidth=2, color='#45B7D1', alpha=0.8)
        
        ax.set_title('Miami-Specific Investments (M$)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Investment (M$)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_demography_international(self, df, ax):
        """Plot de la démographie et des acheteurs internationaux"""
        ax.plot(df['Year'], df['Population']/1000, label='Population', 
               linewidth=2, color='#4ECDC4', alpha=0.8)
        
        ax.set_title('Demography and International Buyers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Population (Thousand)', color='#4ECDC4')
        ax.tick_params(axis='y', labelcolor='#4ECDC4')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(df['Year'], df['International_Buyers_Percentage'], label='International Buyers', 
                linewidth=2, color='#FF6B6B', alpha=0.8)
        ax2.set_ylabel('International Buyers (%)', color='#FF6B6B')
        ax2.tick_params(axis='y', labelcolor='#FF6B6B')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_climate_resilience(self, df, ax):
        """Plot des dépenses climatiques et de résilience"""
        ax.plot(df['Year'], df['Climate_Resilience_Expenses'], label='Climate Resilience', 
               linewidth=2, color='#228B22', alpha=0.8)
        ax.plot(df['Year'], df['Beach_Maintenance_Expenses'], label='Beach Maintenance', 
               linewidth=2, color='#F9A602', alpha=0.8)
        
        ax.set_title('Climate Resilience Expenses (M$)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expenses (M$)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_construction_development(self, df, ax):
        """Plot de la construction et du développement"""
        ax.bar(df['Year'], df['New_Construction_Permits'], label='Construction Permits', 
              color='#45B7D1', alpha=0.7)
        
        ax.set_title('Construction and Development Activity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Construction Permits', color='#45B7D1')
        ax.tick_params(axis='y', labelcolor='#45B7D1')
        ax.grid(True, alpha=0.3, axis='y')
        
        ax2 = ax.twinx()
        ax2.plot(df['Year'], df['Real_Estate_Development'], label='Real Estate Development', 
                linewidth=2, color='#FF6B6B')
        ax2.set_ylabel('Development Investment (M$)', color='#FF6B6B')
        ax2.tick_params(axis='y', labelcolor='#FF6B6B')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_sectorial_investments(self, df, ax):
        """Plot des investissements sectoriels"""
        years = df['Year']
        width = 0.8
        
        bottom = np.zeros(len(years))
        categories = ['Real_Estate_Development', 'Tourism_Infrastructure_Investment', 
                     'Climate_Adaptation_Investment', 'Luxury_Development_Investment',
                     'Marina_Waterfront_Investment']
        
        colors = ['#4ECDC4', '#F9A602', '#228B22', '#FF6B6B', '#45B7D1']
        labels = ['Real Estate Dev', 'Tourism Infra', 'Climate Adaptation', 'Luxury Dev', 'Marina/Waterfront']
        
        for i, category in enumerate(categories):
            ax.bar(years, df[category], width, label=labels[i], bottom=bottom, color=colors[i])
            bottom += df[category]
        
        ax.set_title('Sectorial Investments Distribution (M$)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amount (M$)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _generate_miami_insights(self, df):
        """Génère des insights analytiques adaptés au marché miamien"""
        print(f"🌴 MIAMI/FLORIDA REAL ESTATE INSIGHTS - {self.area}")
        print("=" * 65)
        
        # 1. Statistiques de base
        print("\n1. 📈 KEY STATISTICS:")
        avg_home_price = df['Median_Home_Price'].mean()
        avg_income = df['Median_Income'].mean()
        avg_rent = df['Average_Rent'].mean()
        avg_international_buyers = df['International_Buyers_Percentage'].mean()
        
        print(f"Average median home price: ${avg_home_price:,.0f}")
        print(f"Average median income: ${avg_income:,.0f}")
        print(f"Average rent: ${avg_rent:.0f}")
        print(f"Average international buyers: {avg_international_buyers:.1f}%")
        
        # 2. Croissance immobilière
        print("\n2. 📊 REAL ESTATE GROWTH:")
        price_growth = ((df['Median_Home_Price'].iloc[-1] / 
                        df['Median_Home_Price'].iloc[0]) - 1) * 100
        condo_growth = ((df['Condo_Price_per_Sqft'].iloc[-1] / 
                        df['Condo_Price_per_Sqft'].iloc[0]) - 1) * 100
        
        print(f"Home price growth ({self.start_year}-{self.end_year}): {price_growth:.1f}%")
        print(f"Condo price growth ({self.start_year}-{self.end_year}): {condo_growth:.1f}%")
        print(f"Beachfront premium: {df['Beachfront_Premium'].iloc[-1]:.1f}%")
        
        # 3. Marché international
        print("\n3. 🌍 INTERNATIONAL MARKET:")
        current_international = df['International_Buyers_Percentage'].iloc[-1]
        tourism_revenue = df['Tourism_Tax_Revenue'].iloc[-1]
        
        print(f"Current international buyers: {current_international:.1f}%")
        print(f"Current tourism tax revenue: ${tourism_revenue:.1f}M")
        
        # 4. Accessibilité et marché locatif
        print("\n4. 🏠 HOUSING AFFORDABILITY:")
        current_price = df['Median_Home_Price'].iloc[-1]
        current_income = df['Median_Income'].iloc[-1]
        current_ratio = current_price / current_income
        current_vacancy = df['Rental_Vacancy_Rate'].iloc[-1]
        
        affordability_status = "Critical" if current_ratio > 8 else "Severe" if current_ratio > 6 else "Moderate" if current_ratio > 4 else "Good"
        print(f"Current price-to-income ratio: {current_ratio:.1f} ({affordability_status})")
        print(f"Current rental vacancy rate: {current_vacancy:.1f}%")
        
        # 5. Spécificités de la zone
        print(f"\n5. 🌟 {self.area.upper()} SPECIFICS:")
        print(f"Area type: {self.config['type']}")
        print(f"Specializations: {', '.join(self.config['specialites'])}")
        print(f"Key features: {', '.join(self.config['key_features'])}")
        print(f"Real estate segment: {self.config['segment_immobilier']}")
        
        # 6. Événements marquants du marché floridien
        print("\n6. 📅 KEY FLORIDA REAL ESTATE EVENTS:")
        print("• 2002-2006: Pre-crisis housing bubble and speculation")
        print("• 2007-2011: Subprime crisis - devastating impact in Florida")
        print("• 2005: Hurricane Wilma and active hurricane season")
        print("• 2012-2019: International buyer influx and market recovery")
        print("• 2017: Hurricane Irma and increased climate awareness")
        print("• 2020-2021: COVID-19 pandemic and migration to Florida")
        print("• 2022-present: Insurance crisis and climate adaptation")
        
        # 7. Recommandations stratégiques
        print("\n7. 💡 STRATEGIC RECOMMENDATIONS:")
        if "luxe" in self.config["specialites"]:
            print("• Develop luxury amenities and services")
            print("• Target international high-net-worth buyers")
        if "tourisme" in self.config["specialites"]:
            print("• Enhance tourism infrastructure and experiences")
            print("• Develop year-round tourism offerings")
        if "condos" in self.config["specialites"]:
            print("• Focus on high-rise luxury condo development")
            print("• Invest in building amenities and services")
        
        print("• Increase climate resilience and adaptation measures")
        print("• Address insurance affordability challenges")
        print("• Develop sustainable and green building practices")
        print("• Enhance international marketing and buyer services")
        print("• Invest in public transportation and infrastructure")
        print("• Balance tourism development with residential needs")

def main():
    """Fonction principale pour Miami/Floride"""
    # Liste des zones de Miami/Floride
    areas = ["Miami Downtown", "Miami Beach", "Brickell", "Coral Gables", 
             "Fort Lauderdale", "West Palm Beach", "South Florida Region"]
    
    print("🌴 MIAMI/FLORIDA REAL ESTATE ANALYSIS - KEY AREAS (2002-2025)")
    print("=" * 70)
    
    # Demander à l'utilisateur de choisir une zone
    print("Available areas:")
    for i, area in enumerate(areas, 1):
        print(f"{i}. {area}")
    
    try:
        choice = int(input("\nSelect the area number to analyze: "))
        if choice < 1 or choice > len(areas):
            raise ValueError
        selected_area = areas[choice-1]
    except (ValueError, IndexError):
        print("Invalid choice. Defaulting to Miami Downtown.")
        selected_area = "Miami Downtown"
    
    # Initialiser l'analyseur
    analyzer = MiamiRealEstateAnalyzer(selected_area)
    
    # Générer les données
    real_estate_data = analyzer.generate_financial_data()
    
    # Sauvegarder les données
    output_file = f'{selected_area.replace(" ", "_").lower()}_florida_data_2002_2025.csv'
    real_estate_data.to_csv(output_file, index=False)
    print(f"💾 Data saved: {output_file}")
    
    # Aperçu des données
    print("\n👀 Data preview:")
    print(real_estate_data[['Year', 'Population', 'Median_Home_Price', 'International_Buyers_Percentage', 'Tourism_Tax_Revenue']].head())
    
    # Créer l'analyse
    print("\n📈 Creating Miami/Florida real estate analysis...")
    analyzer.create_financial_analysis(real_estate_data)
    
    print(f"\n✅ Analysis of {selected_area}, Florida completed!")
    print(f"📊 Period: {analyzer.start_year}-{analyzer.end_year}")
    print("🏠 Data: Demographics, real estate market, tourism, international buyers, climate resilience")

if __name__ == "__main__":
    main()