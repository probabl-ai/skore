<script setup lang="ts">
interface Props {
  placeholder?: string;
  type?: string;
  icon?: string;
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: "",
  type: "text",
});

const model = defineModel<string>();

function onChange(event: Event) {
  const input = event.target as HTMLInputElement;
  model.value = input.value;
}
</script>

<template>
  <div class="text-input">
    <span v-if="props.icon" :class="props.icon" class="icon" />
    <input type="text" :value="model" :placeholder="props.placeholder" @keyup="onChange" />
  </div>
</template>

<style scoped>
.text-input {
  display: inline-flex;
  align-items: center;
  padding: var(--spacing-padding-small);
  border: 1px solid var(--border-color-normal);
  border-radius: var(--border-radius);
  background-color: var(--background-color-normal);
  box-shadow: 0 1px 2px var(--background-color-selected);
  color: var(--text-color-normal);

  .icon {
    margin: 0 var(--spacing-gap-normal);
  }

  input {
    width: 100%;
    border: none;
    background-color: transparent;
    color: inherit;
    font-size: var(--text-size-normal);
    font-weight: var(--text-weight-normal);

    &:focus {
      outline: none;
    }
  }

  &:has(input:focus) {
    border-color: var(--color-primary);
    box-shadow: 0 1px 2px rgb(from var(--color-primary) r g b / 20%);
  }
}
</style>
