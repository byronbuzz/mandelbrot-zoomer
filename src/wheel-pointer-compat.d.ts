// Wheel and pointer events both provide the client coordinates used by the shared
// canvas-position helper. This declaration makes that structural compatibility
// explicit for TypeScript's DOM types.
interface WheelEvent extends PointerEvent {}
