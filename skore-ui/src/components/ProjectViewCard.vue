<script setup lang="ts">
import { formatDistance } from "date-fns";
import { onBeforeUnmount, onMounted, ref, useTemplateRef } from "vue";

import DropdownButton from "@/components/DropdownButton.vue";
import DropdownButtonItem from "@/components/DropdownButtonItem.vue";
import FloatingTooltip from "@/components/FloatingTooltip.vue";
import SimpleButton from "@/components/SimpleButton.vue";

const props = defineProps<{
  title: string;
  subtitle?: string;
  showActions: boolean;
  updates?: string[];
  currentUpdateIndex?: number;
}>();

const emit = defineEmits<{
  cardRemoved: [];
  updateSelected: [number];
}>();

const root = useTemplateRef<HTMLDivElement>("root");
const isLatestUpdate = ref(true);

function getUpdateLabel(update: string) {
  const now = new Date();
  return `updated ${formatDistance(update, now)} ago`;
}

function switchToUpdate(index: number) {
  isLatestUpdate.value = index === (props.updates?.length ?? 0) - 1;
  emit("updateSelected", index);
}

function onAnimationEnd() {
  root.value?.classList.remove("blink");
}

function isCurrentlySelectedVersion(index: number) {
  return index === (props.updates?.length ?? 0) - (props.currentUpdateIndex ?? 0) - 1;
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
          <Transition name="fade">
            <span v-if="!isLatestUpdate" class="warning">
              <i class="icon-warning"></i>
              You are viewing an old version
              <a href="#" @click.prevent="switchToUpdate((props.updates?.length ?? 0) - 1)">
                switch to latest
              </a>
            </span>
          </Transition>
        </div>
      </div>
      <div v-if="props.showActions" class="actions">
        <DropdownButton
          icon="icon-history"
          align="right"
          v-if="props.updates && props.updates.length > 1"
        >
          <DropdownButtonItem
            v-for="(item, index) in Array.from(props.updates).reverse()"
            :key="index"
            :icon="isCurrentlySelectedVersion(index) ? 'icon-check' : ''"
            :label="`#${props.updates.length - index} ${getUpdateLabel(item)}`"
            @click="switchToUpdate(props.updates.length - index - 1)"
            icon-position="right"
          />
        </DropdownButton>
        <FloatingTooltip text="Remove from view" placement="bottom-end">
          <SimpleButton icon="icon-trash" @click="emit('cardRemoved')" />
        </FloatingTooltip>
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

        & .warning {
          & a,
          & a:visited {
            color: var(--color-text-secondary);
          }
        }
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
      display: flex;
      flex-direction: row;
      gap: var(--spacing-4);
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
