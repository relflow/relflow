/** Declaration-only schemas shared by the corresponding advanced guide datasets. */
export const repeatPurchaseSchema = `import relflow as rf


class Order(rf.Branch):
    """One order known at the customer's prediction cutoff."""

    amount = rf.Number
    category = rf.Category
    days_ago = rf.Number


class Customer(rf.Model):
    """A customer history and its future-purchase target."""

    days_since_signup = rf.Number
    orders = Order(length=32, overflow="tail")
    returns_in_30d = rf.Boolean(mask=True)`;

export const bearingSchema = `import relflow as rf
from bearing import Bearing


class RouteModel(rf.Model):
    """A known route and its historical travel delay."""

    direction = Bearing
    distance_km = rf.Number
    delay_minutes = rf.Number(mask=True)`;
