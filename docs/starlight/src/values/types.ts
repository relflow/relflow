/** JSON-compatible values shared by nested-value and prediction displays. */
export type Value = string | number | boolean | null | Value[] | { [key: string]: Value };

/** Display-only descriptions keyed by JSON Pointer; never added to copied output. */
export type ValueNotes = Readonly<Record<string, string>>;
