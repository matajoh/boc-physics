"""Module providing the physics system."""

from typing import NamedTuple

from bocpy import Matrix

from .bodies import RigidBody
from .collisions import Collision
from .config import PhysicsMode


ZERO_VEC = Matrix.vector([0, 0])


class PreparedContact(NamedTuple):
    """The per-contact constraint data cached once at the start of a sub-step.

    Description:
        The pose-invariant lever arms and effective mass depend only on the
        bodies' poses, which are fixed for the whole velocity-iteration loop, so
        caching them is a pure hoist. v_target is the restitution velocity bias
        captured from the initial closing speed: it is deliberately sampled once
        here, not per iteration, so the solver drives toward a fixed target.
    """

    ra: Matrix
    rb: Matrix
    ra_perp: Matrix
    rb_perp: Matrix
    ra_perp_dot_n: float
    rb_perp_dot_n: float
    denom: float
    v_target: float


class Constraint(NamedTuple):
    """One prepared pair constraint: the bodies plus their per-contact data.

    Description:
        The output of prepare_collision and the unit the iterate pass applies.
        contacts is empty for the NONE and BASIC modes, which carry no
        pose-invariant data worth caching.
    """

    mode: PhysicsMode
    a: RigidBody
    b: RigidBody
    normal: Matrix
    contacts: tuple


class TangentData(NamedTuple):
    """The fixed friction axis and its per-contact effective mass for a constraint.

    Description:
        Hoisted once per constraint by the solver (the pose is frozen for the
        whole velocity loop) and read every sweep by apply_accumulated. t_axis is
        normal.perpendicular(); denom_t holds the per-contact tangent effective
        mass, so the accumulated friction pass never recomputes pose-invariant
        data inside the iteration loop.
    """

    t_axis: Matrix
    denom_t: tuple


class Physics(NamedTuple):
    """Class providing the physics system.

    Description:
        Note that at several points in this class, we check to see if
        a body participates in the physics system. This is because bodies
        can participate in the collision system but not the physics system.
        It is an immutable NamedTuple so it can ride on the noticeboard and
        be read by every worker sub-interpreter as a shared config snapshot.
    """

    mode: PhysicsMode
    restitution: float = 0.5
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    # closing speed (m/s) below which restitution is treated as zero, so resting
    # contacts settle instead of perpetually micro-bouncing
    restitution_threshold: float = 1.0

    def resolve_collision(self, a: RigidBody, b: RigidBody, collision: Collision,
                          contact0: Matrix, contact1: Matrix):
        """Resolve a collision between two rigid bodies in a single pass."""
        self.apply_collision(self.prepare_collision(a, b, collision, contact0, contact1))

    def prepare_collision(self, a: RigidBody, b: RigidBody, collision: Collision,
                          contact0: Matrix, contact1: Matrix) -> Constraint:
        """Build the pose-invariant constraint for a pair once per sub-step.

        Description:
            The narrow phase fixes the bodies' poses for the whole velocity
            loop, so the lever arms and effective masses are constant across
            iterations. Computing them here, once, lets the iterate pass do only
            the velocity-dependent work. NONE and BASIC carry no such data.
        """
        normal = collision.normal
        if self.mode.is_contact_mode:
            contacts = self.prepare_contacts(a, b, normal, contact0, contact1)
        else:
            contacts = ()

        return Constraint(self.mode, a, b, normal, contacts)

    def prepare_contacts(self, a: RigidBody, b: RigidBody, normal: Matrix,
                         contact0: Matrix, contact1: Matrix) -> tuple:
        """Compute and cache the pose-invariant data for each contact point."""
        points = [contact0]
        if contact1 is not None:
            points.append(contact1)

        prepared = []
        for point in points:
            ra = point - a.position
            rb = point - b.position
            ra_perp = ra.perpendicular()
            rb_perp = rb.perpendicular()
            ra_perp_dot_n = ra_perp.vecdot(normal)
            rb_perp_dot_n = rb_perp.vecdot(normal)
            denom = (a.inv_mass + b.inv_mass +
                     (ra_perp_dot_n * ra_perp_dot_n) * a.inv_inertia +
                     (rb_perp_dot_n * rb_perp_dot_n) * b.inv_inertia)
            v_target = self.restitution_bias(a, b, normal, ra_perp, rb_perp)
            prepared.append(PreparedContact(ra, rb, ra_perp, rb_perp,
                                            ra_perp_dot_n, rb_perp_dot_n, denom,
                                            v_target))

        return tuple(prepared)

    def apply_collision(self, constraint: Constraint):
        """Run one velocity-solver iteration over a prepared constraint."""
        match constraint.mode:
            case PhysicsMode.NONE:
                self.apply_none(constraint.a, constraint.b, constraint.normal)
            case PhysicsMode.BASIC:
                self.apply_basic(constraint.a, constraint.b, constraint.normal)
            case PhysicsMode.ROTATION:
                self.apply_rotation(constraint.a, constraint.b, constraint.normal,
                                    constraint.contacts)
            case PhysicsMode.FRICTION:
                self.apply_friction(constraint.a, constraint.b, constraint.normal,
                                    constraint.contacts)
            case _:
                raise ValueError(f"Invalid physics mode: {constraint.mode}")

    def restitution_for(self, contact_velocity_mag: float) -> float:
        """Return restitution, gated to zero for contacts closing below the threshold."""
        if -contact_velocity_mag > self.restitution_threshold:
            return self.restitution

        return 0.0

    def restitution_bias(self, a: RigidBody, b: RigidBody, normal: Matrix,
                         ra_perp: Matrix, rb_perp: Matrix) -> float:
        """Capture the contact's restitution velocity target from the initial closing speed."""
        va = a.linear_velocity if a.physics else ZERO_VEC
        vb = b.linear_velocity if b.physics else ZERO_VEC
        wva = ra_perp * a.angular_velocity if a.physics else ZERO_VEC
        wvb = rb_perp * b.angular_velocity if b.physics else ZERO_VEC
        vn0 = ((vb + wvb) - (va + wva)).vecdot(normal)
        if -vn0 > self.restitution_threshold:
            return -self.restitution * vn0

        return 0.0

    def apply_none(self, a: RigidBody, b: RigidBody, normal: Matrix):
        """Cancel the normal component of each body's velocity."""
        if a.physics:
            a.linear_velocity -= normal * a.linear_velocity.vecdot(normal)

        if b.physics:
            b.linear_velocity -= normal * b.linear_velocity.vecdot(normal)

    def apply_basic(self, a: RigidBody, b: RigidBody, normal: Matrix):
        """Resolve a collision using impulses without rotation or friction."""
        linear_velocity_a = a.linear_velocity if a.physics else ZERO_VEC
        linear_velocity_b = b.linear_velocity if b.physics else ZERO_VEC
        relative_velocity = linear_velocity_b - linear_velocity_a

        contact_velocity_mag = relative_velocity.vecdot(normal)
        if contact_velocity_mag > 0:
            return

        e = self.restitution_for(contact_velocity_mag)
        j = -(1 + e) * contact_velocity_mag
        j /= a.inv_mass + b.inv_mass
        impulse = j * normal

        if a.physics:
            a.linear_velocity += -impulse * a.inv_mass

        if b.physics:
            b.linear_velocity += impulse * b.inv_mass

    def relative_contact_velocity(self, a: RigidBody, b: RigidBody,
                                  contact: PreparedContact) -> Matrix:
        """Velocity of b relative to a at one contact, including the omega x r terms.

        Description:
            The single closing-velocity formula both the normal and friction
            solves read. Static bodies contribute no linear or angular term. The
            ordering ((b terms) - (a terms)) is preserved verbatim so the extract
            is bit-exact with the three sites it replaces.
        """
        linear_velocity_a = a.linear_velocity if a.physics else ZERO_VEC
        linear_velocity_b = b.linear_velocity if b.physics else ZERO_VEC
        angular_linear_velocity_a = contact.ra_perp * a.angular_velocity if a.physics else ZERO_VEC
        angular_linear_velocity_b = contact.rb_perp * b.angular_velocity if b.physics else ZERO_VEC

        return ((linear_velocity_b + angular_linear_velocity_b) -
                (linear_velocity_a + angular_linear_velocity_a))

    def solve_normal_impulses(self, a: RigidBody, b: RigidBody, normal: Matrix,
                              contacts: tuple) -> list:
        """Run one normal-impulse iteration, applying it and returning each j.

        Description:
            The shared normal solve: drive every contact to its captured
            restitution target, never pulling, then scatter the impulses to the
            two bodies. Returns the per-contact normal magnitude j so the friction
            solve can bound its tangent impulse against it. apply_rotation is just
            this; apply_friction is this plus a tangent pass.
        """
        num_contacts = len(contacts)

        # per-collision scratch is local so concurrent island solves never race
        j_list = [0.0] * num_contacts
        impulse_list = [ZERO_VEC] * num_contacts
        for i in range(num_contacts):
            contact = contacts[i]
            relative_velocity = self.relative_contact_velocity(a, b, contact)

            contact_velocity_mag = relative_velocity.vecdot(normal)
            # drive the contact to its captured restitution target, never pulling
            j = (contact.v_target - contact_velocity_mag)
            j /= contact.denom
            j /= num_contacts
            if j <= 0:
                continue

            j_list[i] = j
            impulse_list[i] = j * normal

        for i in range(num_contacts):
            contact = contacts[i]
            impulse = impulse_list[i]
            if a.physics:
                a.linear_velocity += -impulse * a.inv_mass
                a.angular_velocity += -contact.ra.cross(impulse) * a.inv_inertia

            if b.physics:
                b.linear_velocity += impulse * b.inv_mass
                b.angular_velocity += contact.rb.cross(impulse) * b.inv_inertia

        return j_list

    def apply_rotation(self, a: RigidBody, b: RigidBody, normal: Matrix,
                       contacts: tuple):
        """Resolve a collision using impulses including rotation.

        Description:
            One velocity iteration over the prepared contacts: the lever arms
            and effective masses are read from the prepared data, so only the
            velocity-dependent impulse and its application are computed here. This
            is exactly the shared normal solve, with the returned j discarded.
        """
        self.solve_normal_impulses(a, b, normal, contacts)

    def apply_friction(self, a: RigidBody, b: RigidBody, normal: Matrix,
                       contacts: tuple):
        """Resolve a collision using impulses including rotation and friction.

        Description:
            One velocity iteration over the prepared contacts. The normal solve is
            the shared solve_normal_impulses; the tangent direction and its
            effective mass depend on the post-normal velocity, so they are
            recomputed here each iteration and bounded by the Coulomb cone.
        """
        num_contacts = len(contacts)
        sf = self.static_friction
        df = self.dynamic_friction

        j_list = self.solve_normal_impulses(a, b, normal, contacts)

        # per-collision scratch is local so concurrent island solves never race
        friction_impulse_list = [ZERO_VEC] * num_contacts

        for i in range(num_contacts):
            contact = contacts[i]
            j = j_list[i]

            relative_velocity = self.relative_contact_velocity(a, b, contact)

            tangent = relative_velocity - relative_velocity.vecdot(normal) * normal

            if tangent.magnitude_squared() < 1e-5:
                continue

            tangent = tangent.normalize()
            ra_perp_dot_t = contact.ra_perp.vecdot(tangent)
            rb_perp_dot_t = contact.rb_perp.vecdot(tangent)

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
            contact = contacts[i]
            friction_impulse = friction_impulse_list[i]
            if a.physics:
                a.linear_velocity += -friction_impulse * a.inv_mass
                a.angular_velocity += -contact.ra.cross(friction_impulse) * a.inv_inertia

            if b.physics:
                b.linear_velocity += friction_impulse * b.inv_mass
                b.angular_velocity += contact.rb.cross(friction_impulse) * b.inv_inertia

    def apply_accumulated(self, constraint: Constraint, lam_n: list, lam_t: list,
                          tangent_data: TangentData):
        """Run one accumulated-PGS sweep over a constraint, clamping running totals.

        Description:
            Unlike apply_friction, which clamps each iteration's increment, this
            clamps the RUNNING total impulse (lam_n, lam_t) and applies only the
            change, so a later sweep can release earlier over-push -- the cure for
            the creep that equal-budget per-iteration clamping leaves in. The
            normal and two-coefficient friction passes each use the stock
            per-manifold block solve (read every contact against the shared
            velocity, then scatter together), so contact order within a manifold
            never biases the result.
        """
        a = constraint.a
        b = constraint.b
        normal = constraint.normal
        contacts = constraint.contacts
        num_contacts = len(contacts)
        if num_contacts == 0:
            return

        normal_applied = [ZERO_VEC] * num_contacts
        for i in range(num_contacts):
            contact = contacts[i]
            relative_velocity = self.relative_contact_velocity(a, b, contact)
            contact_velocity_mag = relative_velocity.vecdot(normal)
            # v_target was sampled once at prepare time; the fixed point is still
            # vn -> v_target, so restitution matches apply_friction exactly
            delta = (contact.v_target - contact_velocity_mag) / contact.denom / num_contacts
            new_total = lam_n[i] + delta
            if new_total < 0.0:
                new_total = 0.0

            normal_applied[i] = (new_total - lam_n[i]) * normal
            lam_n[i] = new_total

        self.scatter_impulses(a, b, contacts, normal_applied)
        self.accumulated_friction(a, b, contacts, lam_n, lam_t, tangent_data)

    def accumulated_friction(self, a: RigidBody, b: RigidBody, contacts: tuple,
                             lam_n: list, lam_t: list, tangent_data: TangentData):
        """Accumulate the tangent impulse along the fixed axis, two-coefficient Coulomb.

        Description:
            The running tangent total sticks while inside the static cone
            (+/- static_friction * lam_n) and otherwise slides, capped at the
            kinetic cone (+/- dynamic_friction * lam_n) -- the same two-coefficient
            model as apply_friction, expressed on the accumulated total. The axis
            is fixed, so there is no tangent.magnitude_squared() < 1e-5 guard: the
            applied delta is ~0 at zero tangential speed.
        """
        num_contacts = len(contacts)
        t_axis = tangent_data.t_axis
        denom_t = tangent_data.denom_t
        sf = self.static_friction
        df = self.dynamic_friction
        friction_applied = [ZERO_VEC] * num_contacts
        for i in range(num_contacts):
            contact = contacts[i]
            relative_velocity = self.relative_contact_velocity(a, b, contact)
            jt = -relative_velocity.vecdot(t_axis) / denom_t[i] / num_contacts
            new_total = lam_t[i] + jt
            static_bound = sf * lam_n[i]
            if new_total > static_bound or new_total < -static_bound:
                # outside the static cone -> slide: cap the running total at kinetic
                kinetic_bound = df * lam_n[i]
                if new_total > kinetic_bound:
                    new_total = kinetic_bound
                elif new_total < -kinetic_bound:
                    new_total = -kinetic_bound

            friction_applied[i] = (new_total - lam_t[i]) * t_axis
            lam_t[i] = new_total

        self.scatter_impulses(a, b, contacts, friction_applied)

    def scatter_impulses(self, a: RigidBody, b: RigidBody, contacts: tuple,
                         impulses: list):
        """Apply per-contact impulses to the two bodies, after the read pass.

        Description:
            The write half of the per-manifold block solve: every impulse was
            computed against the same shared velocity in the caller's read loop,
            so applying them here, after that loop, keeps a manifold's contact
            points order-independent (a block update, not point-by-point
            Gauss-Seidel within the manifold).
        """
        for i in range(len(contacts)):
            contact = contacts[i]
            impulse = impulses[i]
            if a.physics:
                a.linear_velocity += -impulse * a.inv_mass
                a.angular_velocity += -contact.ra.cross(impulse) * a.inv_inertia

            if b.physics:
                b.linear_velocity += impulse * b.inv_mass
                b.angular_velocity += contact.rb.cross(impulse) * b.inv_inertia
