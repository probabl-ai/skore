<script setup lang="ts">
import { onMounted, ref } from "vue";

interface Props {
  placeholder?: string;
  type?: string;
  icon?: string;
  focus?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: "",
  type: "text",
});

const model = defineModel<string>();
const input = ref<HTMLInputElement | null>(null);

function onChange(event: Event) {
  const input = event.target as HTMLInputElement;
  model.value = input.value;
}

onMounted(() => {
  if (props.focus && input.value) {
    input.value.focus();
  }
});
</script>

<template>
  <div class="text-input">
    <span v-if="props.icon" :class="props.icon" class="icon" />
    <input
      type="text"
      :value="model"
      :placeholder="props.placeholder"
      @keyup="onChange"
      ref="input"
    />
  </div>
</template>

<style scoped>
.text-input {
  display: inline-flex;
  align-items: center;
  padding: var(--spacing-8);
  border: var(--stroke-width-md) solid var(--color-stroke-background-primary);
  border-radius: var(--radius-xs);
  box-shadow: 0 1px 2px var(--shadow-color);
  transition: all var(--animation-duration) var(--animation-easing);

  .icon {
    margin: 0 var(--spacing-10);
    color: var(--color-text-primary);
  }

  input {
    width: 100%;
    min-width: 0;
    border: none;
    background-color: transparent;
    color: var(--color-text-primary);

    &:focus {
      outline: none;
    }

    &::placeholder {
      color: var(--color-text-secondary);
    }
  }

  &:has(input:focus) {
    border-color: var(--color-stroke-background-primary);
    box-shadow: 0 1px 2px rgb(from var(--color-primary) r g b / 20%);
  }
}
</style>
