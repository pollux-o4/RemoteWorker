// Minimal DOM host for interaction contracts; native browser layout/focus QA remains separate.
export function installProfileDom() {
    class Element {
        constructor(tag) {
            this.tagName = tag.toUpperCase(); this.children = []; this.style = {}; this.dataset = {};
            this.listeners = new Map(); this.attributes = new Map(); this.disabled = false; this.hidden = false;
            this._value = ""; this.textContent = ""; this.validationMessage = ""; this.open = false;
        }
        append(...children) {
            for (const child of children) {
                if (child.parent) child.remove();
                this.children.push(child); child.parent = this;
            }
        }
        replaceChildren(...children) { for (const child of this.children) child.parent = null; this.children = []; this.append(...children); }
        remove() { if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); this.parent = null; }
        get options() { return this.children.filter(child => child.tagName === "OPTION"); }
        set value(value) { this._value = String(value ?? ""); }
        get value() { return this._value; }
        setAttribute(name, value) { this.attributes.set(name, value); }
        removeAttribute(name) { this.attributes.delete(name); }
        getAttribute(name) { return this.attributes.get(name); }
        addEventListener(name, callback, options) { const listeners = this.listeners.get(name) || []; listeners.push({ callback, options }); this.listeners.set(name, listeners); }
        dispatch(name, overrides = {}) {
            const event = { defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...overrides };
            for (const entry of [...this.listeners.get(name) || []]) { entry.callback(event); if (entry.options?.once) this.listeners.set(name, this.listeners.get(name).filter(item => item !== entry)); }
            return event;
        }
        click() { if (!this.disabled) this.dispatch("click"); }
        focus() { document.activeElement = this; }
        showModal() { this.open = true; this.focus(); }
        close() { if (!this.open) return; this.open = false; this.dispatch("close"); }
        setCustomValidity(value) { this.validationMessage = value; }
        checkValidity() {
            if (this.disabled) return true;
            return !this.validationMessage && (!this.required || Boolean(this.value)) &&
                (!this.pattern || new RegExp(`^(?:${this.pattern})$`).test(this.value)) &&
                (this.type !== "number" || (Number.isInteger(Number(this.value)) && Number(this.value) >= Number(this.min) && Number(this.value) <= Number(this.max)));
        }
        reportValidity() { const invalid = [this, ...this.querySelectorAll("input, select, textarea")].find(control => !control.checkValidity()); invalid?.focus(); return !invalid; }
        querySelectorAll(selector) {
            const descendants = this.children.flatMap(child => [child, ...child.querySelectorAll("*")]);
            if (selector === "*") return descendants;
            return descendants.filter(child => selector.split(",").some(part => {
                const tag = part.trim().split(":")[0].toUpperCase();
                return child.tagName === tag && (!part.includes(":not") || child.type !== "checkbox");
            }));
        }
    }
    globalThis.document = { activeElement: null, createElement: tag => new Element(tag), createTextNode: text => Object.assign(new Element("#text"), { textContent: text }), body: new Element("body") };
    globalThis.window = { location: { origin: "http://local" } };
    const role = (dialog, name) => [dialog, ...dialog.querySelectorAll("*")].find(element => element.dataset.profileRole === name);
    return { role, Element };
}
