"""Synthetic customers with known fraud injected, so results can be measured."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .engine import Transaction

START = datetime(2026, 1, 1, 9, 0)


@dataclass(frozen=True)
class Archetype:
    name: str
    median_amount: float
    sigma: float
    active_hours: tuple[int, ...]
    home_country: str
    daily_rate: float


ARCHETYPES = (
    Archetype("student", 60, 0.55, tuple(range(8, 24)) + (0, 1), "TR", 1.6),
    Archetype("salaried", 250, 0.5, tuple(range(7, 23)), "TR", 1.1),
    Archetype("business", 2500, 0.7, tuple(range(0, 24)), "TR", 2.4),
    Archetype("retiree", 120, 0.45, tuple(range(8, 20)), "TR", 0.7),
)

MERCHANTS = ("Migros", "Shell", "Trendyol", "Getir", "BIM", "Starbucks", "Hepsiburada")
FOREIGN = ("DE", "NL", "GB", "AE")


@dataclass(frozen=True)
class LabelledTransaction:
    transaction: Transaction
    is_fraud: bool
    episode: str


@dataclass
class SimulatedCustomer:
    customer_id: str
    archetype: Archetype
    device: str
    rng: random.Random
    countries: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.countries = [self.archetype.home_country]
        if self.archetype.name == "business":
            self.countries += list(self.rng.sample(FOREIGN, 2))

    def legit_amount(self) -> float:
        mu = math.log(self.archetype.median_amount)
        return max(round(self.rng.lognormvariate(mu, self.archetype.sigma), 2), 1.0)

    def legit_transaction(self, when: datetime) -> LabelledTransaction:
        hour = self.rng.choice(self.archetype.active_hours)
        stamp = when.replace(hour=hour, minute=self.rng.randrange(60), second=0, microsecond=0)
        country = self.rng.choices(
            self.countries, weights=[8] + [1] * (len(self.countries) - 1)
        )[0]

        return LabelledTransaction(
            transaction=Transaction(
                customer_id=self.customer_id,
                amount=self.legit_amount(),
                timestamp=stamp,
                device_id=self.device,
                country=country,
                merchant=self.rng.choice(MERCHANTS),
            ),
            is_fraud=False,
            episode="legit",
        )


    def account_takeover(self, when: datetime) -> list[LabelledTransaction]:
        """Stolen credentials: unfamiliar device and country, draining fast."""
        device = f"stolen-{self.rng.randrange(1000)}"
        country = self.rng.choice([c for c in FOREIGN if c not in self.countries] or ["RU"])
        stamp = when.replace(hour=self.rng.choice((2, 3, 4)), minute=0)

        out = []
        for i in range(self.rng.randint(2, 4)):
            out.append(LabelledTransaction(
                transaction=Transaction(
                    customer_id=self.customer_id,
                    amount=round(self.archetype.median_amount * self.rng.uniform(8, 25), 2),
                    timestamp=stamp + timedelta(minutes=3 * i),
                    device_id=device,
                    country=country,
                    merchant="Transfer",
                ),
                is_fraud=True,
                episode="takeover",
            ))
        return out

    def card_testing(self, when: datetime) -> list[LabelledTransaction]:
        """Validating a stolen card with a rapid burst of tiny charges."""
        device = f"tester-{self.rng.randrange(1000)}"
        stamp = when.replace(hour=self.rng.randrange(24), minute=0)

        return [
            LabelledTransaction(
                transaction=Transaction(
                    customer_id=self.customer_id,
                    amount=round(self.rng.uniform(1, 9), 2),
                    timestamp=stamp + timedelta(minutes=i),
                    device_id=device,
                    country=self.rng.choice(FOREIGN),
                    merchant="Online",
                ),
                is_fraud=True,
                episode="card_testing",
            )
            for i in range(self.rng.randint(4, 7))
        ]

    def escalation(self, when: datetime) -> list[LabelledTransaction]:
        """The patient attack: start small from a new device, climb steadily."""
        device = f"creep-{self.rng.randrange(1000)}"
        out = []
        amount = self.archetype.median_amount * 1.5
        for i in range(8):
            stamp = (when + timedelta(days=i)).replace(
                hour=self.rng.choice(self.archetype.active_hours), minute=0
            )
            out.append(LabelledTransaction(
                transaction=Transaction(
                    customer_id=self.customer_id,
                    amount=round(amount, 2),
                    timestamp=stamp,
                    device_id=device,
                    country=self.archetype.home_country,
                    merchant="Marketplace",
                ),
                is_fraud=True,
                episode="escalation",
            ))
            amount *= 1.8
        return out


def generate_stream(
    seed: int = 20260810,
    customers: int = 60,
    days: int = 120,
    fraud_rate: float = 0.18,
) -> list[LabelledTransaction]:
    """A time-ordered stream of labelled transactions. Fraud only starts once a customer has history."""
    rng = random.Random(seed)
    people = [
        SimulatedCustomer(
            customer_id=f"cust-{i:03d}",
            archetype=ARCHETYPES[i % len(ARCHETYPES)],
            device=f"device-{i:03d}",
            rng=random.Random(seed + i),
        )
        for i in range(customers)
    ]

    stream: list[LabelledTransaction] = []

    for person in people:
        for day in range(days):
            when = START + timedelta(days=day)
            count = 1 if person.rng.random() < person.archetype.daily_rate % 1 else 0
            count += int(person.archetype.daily_rate)
            for _ in range(count):
                stream.append(person.legit_transaction(when))

        if rng.random() < fraud_rate:
            attack_day = rng.randrange(days // 3, days - 10)
            when = START + timedelta(days=attack_day)
            episode = rng.choice(("takeover", "card_testing", "escalation"))
            if episode == "takeover":
                stream.extend(person.account_takeover(when))
            elif episode == "card_testing":
                stream.extend(person.card_testing(when))
            else:
                stream.extend(person.escalation(when))

    stream.sort(key=lambda item: item.transaction.timestamp)
    return stream
