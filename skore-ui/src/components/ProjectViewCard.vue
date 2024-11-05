<script setup lang="ts">
import DropdownButton from "@/components/DropdownButton.vue";
import DropdownButtonItem from "@/components/DropdownButtonItem.vue";
import { onBeforeUnmount, onMounted, useTemplateRef } from "vue";

const props = defineProps<{
  title: string;
  subtitle?: string;
  showActions: boolean;
}>();

const emit = defineEmits<{
  cardRemoved: [];
}>();

const root = useTemplateRef<HTMLDivElement>("root");

function onAnimationEnd() {
  root.value?.classList.remove("blink");
}

onMounted(() => {
  root.value?.addEventListener("animationend", onAnimationEnd);
});

onBeforeUnmount(() => {
  root.value?.removeEventListener("animationend", onAnimationEnd);
});
</script>

<template>
  <div class="card" ref="root">
    <div class="header">
      <div class="titles">
        <div class="title">{{ props.title }}</div>
        <div class="subtitle" v-if="props.subtitle">
          {{ props.subtitle }}
        </div>
      </div>
      <div v-if="props.showActions" class="actions">
        <DropdownButton icon="icon-more" align="right">
          <DropdownButtonItem
            label="Remove from view"
            icon="icon-trash"
            @click="emit('cardRemoved')"
          />
        </DropdownButton>
      </div>
    </div>
    <hr />
    <div class="content">
      <slot></slot>
    </div>
  </div>
</template>

<style scoped>
.card {
  position: relative;
  overflow: auto;
  max-width: 100%;
  background-color: var(--color-background-primary);

  & .header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;

    & .titles {
      position: relative;
      padding-left: calc(var(--spacing-8) + 4px);

      & .title {
        font-size: var(--font-size-sm);
      }

      & .subtitle {
        color: var(--color-text-secondary);
        font-size: var(--font-size-xs);
      }

      &::before {
        position: absolute;
        top: 0;
        left: 0;
        display: block;
        width: 4px;
        height: 100%;
        border-radius: var(--radius-xs);
        background-color: var(--color-background-branding);
        content: "";
      }
    }

    & .actions {
      opacity: 0;
      transition: opacity var(--animation-duration) var(--animation-easing);
    }
  }

  & hr {
    border: none;
    border-top: solid var(--stroke-width-md) var(--color-stroke-background-primary);
    margin: var(--spacing-16) 0;
  }

  &:hover {
    & .header .actions {
      opacity: 1;
    }
  }

  &.blink {
    & .header {
      & .titles {
        &::before {
          animation-duration: var(--animation-duration);
          animation-iteration-count: 5;
          animation-name: blink;
          animation-play-state: running;
          animation-timing-function: var(--animation-easing);
        }
      }
    }
  }
}

@keyframes blink {
  0% {
    background-color: var(--color-background-branding);
  }

  50% {
    background-color: var(--color-background-primary);
  }

  100% {
    background-color: var(--color-background-branding);
  }
}
</style>
