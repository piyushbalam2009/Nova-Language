import { GradVar, NovaOptimizer, NovaTensor, SequentialModel } from './ai';

export type NovaValue =
  | { kind: 'Int'; value: number }
  | { kind: 'Float'; value: number }
  | { kind: 'Bool'; value: boolean }
  | { kind: 'Char'; value: string }
  | { kind: 'None' }
  | { kind: 'HeapRef'; id: number };

export type HeapPayload =
  | { tag: 'String'; value: string }
  | { tag: 'Array'; elements: NovaValue[] }
  | { tag: 'Map'; entries: Map<string, NovaValue> }
  | { tag: 'Struct'; typeName: string; fields: Map<string, NovaValue> }
  | { tag: 'Result'; variant: 'Ok' | 'Error'; value: NovaValue }
  | { tag: 'Tensor'; tensor: NovaTensor }
  | { tag: 'GradVar'; gradVar: GradVar }
  | { tag: 'Model'; model: SequentialModel }
  | { tag: 'Optimizer'; optimizer: NovaOptimizer };

export interface HeapObject {
  id: number;
  payload: HeapPayload;
  marked: boolean;
  generation: number; // Designed for future generational GC promotion
  sizeBytes: number;
}

export interface GCStats {
  totalAllocated: number;
  totalFreed: number;
  liveObjects: number;
  liveBytes: number;
  collectionsRun: number;
}

export class NovaHeap {
  private nextId = 1;
  private objects = new Map<number, HeapObject>();
  private totalAllocated = 0;
  private totalFreed = 0;
  private collectionsRun = 0;
  public gcThreshold = 64;

  alloc(payload: HeapPayload): NovaValue {
    const id = this.nextId++;
    const sizeBytes = this.estimateBytes(payload);
    this.objects.set(id, {
      id,
      payload,
      marked: false,
      generation: 0,
      sizeBytes,
    });
    this.totalAllocated++;
    return { kind: 'HeapRef', id };
  }

  deref(val: NovaValue): HeapObject {
    if (val.kind !== 'HeapRef') {
      throw new Error(`Expected heap reference, got ${val.kind}`);
    }
    const obj = this.objects.get(val.id);
    if (!obj) {
      throw new Error(`Dangling heap reference #${val.id}`);
    }
    return obj;
  }

  shouldCollect(): boolean {
    return this.objects.size >= this.gcThreshold;
  }

  /**
   * Mark-and-sweep garbage collector.
   * Marks all reachable heap objects from roots (operand stack, call frame locals, globals),
   * then sweeps unreachable objects and promotes surviving objects' generation counter.
   */
  collect(roots: NovaValue[]): number {
    this.collectionsRun++;
    for (const obj of this.objects.values()) {
      obj.marked = false;
    }

    const worklist: number[] = [];
    for (const root of roots) {
      if (root.kind === 'HeapRef') worklist.push(root.id);
    }

    while (worklist.length > 0) {
      const id = worklist.pop()!;
      const obj = this.objects.get(id);
      if (!obj || obj.marked) continue;
      obj.marked = true;
      obj.generation++;

      const p = obj.payload;
      if (p.tag === 'Array') {
        for (const el of p.elements) {
          if (el.kind === 'HeapRef') worklist.push(el.id);
        }
      } else if (p.tag === 'Map') {
        for (const v of p.entries.values()) {
          if (v.kind === 'HeapRef') worklist.push(v.id);
        }
      } else if (p.tag === 'Struct') {
        for (const v of p.fields.values()) {
          if (v.kind === 'HeapRef') worklist.push(v.id);
        }
      } else if (p.tag === 'Result') {
        if (p.value.kind === 'HeapRef') worklist.push(p.value.id);
      }
    }

    let freed = 0;
    for (const [id, obj] of this.objects.entries()) {
      if (!obj.marked) {
        this.objects.delete(id);
        freed++;
        this.totalFreed++;
      }
    }

    this.gcThreshold = Math.max(64, this.objects.size * 2);
    return freed;
  }

  getStats(): GCStats {
    let liveBytes = 0;
    for (const obj of this.objects.values()) liveBytes += obj.sizeBytes;
    return {
      totalAllocated: this.totalAllocated,
      totalFreed: this.totalFreed,
      liveObjects: this.objects.size,
      liveBytes,
      collectionsRun: this.collectionsRun,
    };
  }

  formatValue(val: NovaValue): string {
    switch (val.kind) {
      case 'Int': return String(val.value);
      case 'Float': return Number.isInteger(val.value) ? `${val.value}.0` : String(val.value);
      case 'Bool': return val.value ? 'true' : 'false';
      case 'Char': return val.value;
      case 'None': return 'None';
      case 'HeapRef': {
        const obj = this.objects.get(val.id);
        if (!obj) return '<freed>';
        const p = obj.payload;
        switch (p.tag) {
          case 'String': return p.value;
          case 'Array': return `[${p.elements.map((e) => this.formatValue(e)).join(', ')}]`;
          case 'Map': {
            const parts: string[] = [];
            for (const [k, v] of p.entries.entries()) {
              parts.push(`${JSON.stringify(k)}: ${this.formatValue(v)}`);
            }
            return `{${parts.join(', ')}}`;
          }
          case 'Struct': {
            const parts: string[] = [];
            for (const [k, v] of p.fields.entries()) {
              parts.push(`${k}: ${this.formatValue(v)}`);
            }
            return `${p.typeName} { ${parts.join(', ')} }`;
          }
          case 'Result':
            return `${p.variant}(${this.formatValue(p.value)})`;
          case 'Tensor':
            return p.tensor.toString();
          case 'GradVar':
            return `GradVar(data=${p.gradVar.data.toString()}, grad=${p.gradVar.grad.toString()})`;
          case 'Model':
            return `Sequential(layers=${p.model.layers.length}, params=${p.model.paramCount()})`;
          case 'Optimizer':
            return `Optimizer(${p.optimizer.name})`;
        }
      }
    }
  }

  private estimateBytes(payload: HeapPayload): number {
    switch (payload.tag) {
      case 'String': return 24 + payload.value.length * 2;
      case 'Array': return 32 + payload.elements.length * 16;
      case 'Map': return 48 + payload.entries.size * 32;
      case 'Struct': return 32 + payload.fields.size * 24;
      case 'Result': return 24;
      case 'Tensor': return 48 + payload.tensor.data.byteLength;
      case 'GradVar': return 96 + payload.gradVar.data.data.byteLength * 2;
      case 'Model': return 128 + payload.model.paramCount() * 16;
      case 'Optimizer': return 128;
    }
  }
}
