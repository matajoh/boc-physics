"""Module providing the physics system."""

from bocpy import Matrix

from .bodies import RigidBody
from .collisions import Collision
from .config import PhysicsMode


ZERO_VEC = Matrix.vector([0, 0])


class Physics:
    """Class providing the physics system.

    Description:
        Note that at several points in this class, we check to see if
        a body participates in the physics system. This is because bodies
        can participate in the collision system but not the physics system.
    """

    def __init__(self, mode: PhysicsMode,
                 restitution: float = 0.5,
                 static_friction: float = 0.5,
                 dynamic_friction: float = 0.5):
        """Create the physics system from its mode and material coefficients."""
        self.mode = mode
        self.restitution = restitution
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction

    def resolve_collision(self, a: RigidBody, b: RigidBody, collision: Collision,
                          contact0: Matrix, contact1: Matrix):
        """Resolve a collision between two rigid bodies."""
        normal = collision.normal
        match self.mode:
            case PhysicsMode.NONE:
                self.resolve_collision_none(a, b, normal)
            case PhysicsMode.BASIC:
                self.resolve_collision_basic(a, b, normal)
            case PhysicsMode.ROTATION:
                self.resolve_collision_rotation(a, b, normal, contact0, contact1)
            case PhysicsMode.FRICTION:
                self.resolve_collision_friction(a, b, normal, contact0, contact1)
            case _:
                raise ValueError(f"Invalid physics mode: {self.mode}")

    def resolve_collision_none(self, a: RigidBody, b: RigidBody, normal: Matrix):
        """Cancel the normal component of each body's velocity."""
        if a.physics:
            a.linear_velocity -= normal * a.linear_velocity.vecdot(normal)

        if b.physics:
            b.linear_velocity -= normal * b.linear_velocity.vecdot(normal)

    def resolve_collision_basic(self, a: RigidBody, b: RigidBody, normal: Matrix):
        """Resolve a collision using impulses without rotation or friction."""
        linear_velocity_a = a.linear_velocity if a.physics else ZERO_VEC
        linear_velocity_b = b.linear_velocity if b.physics else ZERO_VEC
        relative_velocity = linear_velocity_b - linear_velocity_a

        contact_velocity_mag = relative_velocity.vecdot(normal)
        if contact_velocity_mag > 0:
            return

        e = self.restitution
        j = -(1 + e) * contact_velocity_mag
        j /= a.inv_mass + b.inv_mass
        impulse = j * normal

        if a.physics:
            a.linear_velocity += -impulse * a.inv_mass

        if b.physics:
            b.linear_velocity += impulse * b.inv_mass

    def resolve_collision_rotation(self, a: RigidBody, b: RigidBody, normal: Matrix,
                                   contact0: Matrix, contact1: Matrix):
        """Resolve a collision using impulses including rotation."""
        contacts = [contact0]
        if contact1 is not None:
            contacts.append(contact1)

        num_contacts = len(contacts)
        e = self.restitution

        # per-collision scratch is local so concurrent island solves never race
        ra_list = [ZERO_VEC] * num_contacts
        rb_list = [ZERO_VEC] * num_contacts
        impulse_list = [ZERO_VEC] * num_contacts

        for i in range(num_contacts):
            ra = contacts[i] - a.position
            rb = contacts[i] - b.position
            ra_list[i] = ra
            rb_list[i] = rb

            ra_perp = ra.perpendicular()
            rb_perp = rb.perpendicular()

            linear_velocity_a = a.linear_velocity if a.physics else ZERO_VEC
            linear_velocity_b = b.linear_velocity if b.physics else ZERO_VEC
            angular_linear_velocity_a = ra_perp * a.angular_velocity if a.physics else ZERO_VEC
            angular_linear_velocity_b = rb_perp * b.angular_velocity if b.physics else ZERO_VEC

            relative_velocity = ((linear_velocity_b + angular_linear_velocity_b) -
                                 (linear_velocity_a + angular_linear_velocity_a))

            contact_velocity_mag = relative_velocity.vecdot(normal)
            if contact_velocity_mag > 0:
                continue

            ra_perp_dot_n = ra_perp.vecdot(normal)
            rb_perp_dot_n = rb_perp.vecdot(normal)

            denom = (a.inv_mass + b.inv_mass +
                     (ra_perp_dot_n * ra_perp_dot_n) * a.inv_inertia +
                     (rb_perp_dot_n * rb_perp_dot_n) * b.inv_inertia)

            j = -(1 + e) * contact_velocity_mag
            j /= denom
            j /= num_contacts
            impulse = j * normal

            impulse_list[i] = impulse

        for i in range(num_contacts):
            impulse = impulse_list[i]
            ra = ra_list[i]
            rb = rb_list[i]
            if a.physics:
                a.linear_velocity += -impulse * a.inv_mass
                a.angular_velocity += -ra.cross(impulse) * a.inv_inertia

            if b.physics:
                b.linear_velocity += impulse * b.inv_mass
                b.angular_velocity += rb.cross(impulse) * b.inv_inertia

    def resolve_collision_friction(self, a: RigidBody, b: RigidBody, normal: Matrix,
                                   contact0: Matrix, contact1: Matrix):
        """Resolve a collision using impulses including rotation and friction."""
        contacts = [contact0]
        if contact1 is not None:
            contacts.append(contact1)

        num_contacts = len(contacts)
        e = self.restitution
        sf = self.static_friction
        df = self.dynamic_friction

        # per-collision scratch is local so concurrent island solves never race
        ra_list = [ZERO_VEC] * num_contacts
        rb_list = [ZERO_VEC] * num_contacts
        j_list = [0] * num_contacts
        impulse_list = [ZERO_VEC] * num_contacts
        friction_impulse_list = [ZERO_VEC] * num_contacts

        for i in range(num_contacts):
            ra = contacts[i] - a.position
            rb = contacts[i] - b.position
            ra_list[i] = ra
            rb_list[i] = rb

            ra_perp = ra.perpendicular()
            rb_perp = rb.perpendicular()

            linear_velocity_a = a.linear_velocity if a.physics else ZERO_VEC
            linear_velocity_b = b.linear_velocity if b.physics else ZERO_VEC
            angular_linear_velocity_a = ra_perp * a.angular_velocity if a.physics else ZERO_VEC
            angular_linear_velocity_b = rb_perp * b.angular_velocity if b.physics else ZERO_VEC

            relative_velocity = ((linear_velocity_b + angular_linear_velocity_b) -
                                 (linear_velocity_a + angular_linear_velocity_a))

            contact_velocity_mag = relative_velocity.vecdot(normal)
            if contact_velocity_mag > 0:
                continue

            ra_perp_dot_n = ra_perp.vecdot(normal)
            rb_perp_dot_n = rb_perp.vecdot(normal)

            denom = (a.inv_mass + b.inv_mass +
                     (ra_perp_dot_n * ra_perp_dot_n) * a.inv_inertia +
                     (rb_perp_dot_n * rb_perp_dot_n) * b.inv_inertia)

            j = -(1 + e) * contact_velocity_mag
            j /= denom
            j /= num_contacts
            j_list[i] = j
            impulse = j * normal
            impulse_list[i] = impulse

        for i in range(num_contacts):
            impulse = impulse_list[i]
            ra = ra_list[i]
            rb = rb_list[i]
            if a.physics:
                a.linear_velocity += -impulse * a.inv_mass
                a.angular_velocity += -ra.cross(impulse) * a.inv_inertia

            if b.physics:
                b.linear_velocity += impulse * b.inv_mass
                b.angular_velocity += rb.cross(impulse) * b.inv_inertia

        for i in range(num_contacts):
            ra = ra_list[i]
            rb = rb_list[i]
            j = j_list[i]

            ra_perp = ra.perpendicular()
            rb_perp = rb.perpendicular()

            linear_velocity_a = a.linear_velocity if a.physics else ZERO_VEC
            linear_velocity_b = b.linear_velocity if b.physics else ZERO_VEC
            angular_linear_velocity_a = ra_perp * a.angular_velocity if a.physics else ZERO_VEC
            angular_linear_velocity_b = rb_perp * b.angular_velocity if b.physics else ZERO_VEC

            relative_velocity = ((linear_velocity_b + angular_linear_velocity_b) -
                                 (linear_velocity_a + angular_linear_velocity_a))

            tangent = relative_velocity - relative_velocity.vecdot(normal) * normal

            if tangent.magnitude_squared() < 1e-5:
                continue

            tangent = tangent.normalize()
            ra_perp_dot_t = ra_perp.vecdot(tangent)
            rb_perp_dot_t = rb_perp.vecdot(tangent)

            denom = (a.inv_mass + b.inv_mass +
                     (ra_perp_dot_t * ra_perp_dot_t) * a.inv_inertia +
                     (rb_perp_dot_t * rb_perp_dot_t) * b.inv_inertia)

            jt = -relative_velocity.vecdot(tangent)
            jt /= denom
            jt /= num_contacts

            if abs(jt) <= j * sf:
                friction_impulse = jt * tangent
            else:
                friction_impulse = -j * tangent * df

            friction_impulse_list[i] = friction_impulse

        for i in range(num_contacts):
            friction_impulse = friction_impulse_list[i]
            ra = ra_list[i]
            rb = rb_list[i]
            if a.physics:
                a.linear_velocity += -friction_impulse * a.inv_mass
                a.angular_velocity += -ra.cross(friction_impulse) * a.inv_inertia

            if b.physics:
                b.linear_velocity += friction_impulse * b.inv_mass
                b.angular_velocity += rb.cross(friction_impulse) * b.inv_inertia
